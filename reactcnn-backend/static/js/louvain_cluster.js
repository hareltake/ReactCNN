const width_c = 600;
const height_c = 500;

const padding = 0;
const clusterPadding = 20;

const maxRadius = 4;

const z = d3.scaleOrdinal(d3.schemeCategory20);

var clusters = {};
const defaultRadius = 4;
var last_result = {}

function louvain_cluster(layer) {
    d3.json("http://127.0.0.1:5000/getCorrData?l=" + layer, function(error, graph) {
            var filters_num = Object.keys(graph).length;
            nodes = new Array();
            nodeData = new Array();
            linkData = new Array();
            for (var i = 0; i < filters_num; i++) {
                nodeData[i] = i;
                node = {id: i};
                nodes[i] = node;
            }

            var i = 0;
            for (var x = 0; x < filters_num; x++) {
                for (var y = x + 1; y < filters_num; y++){
                    link = {source: x, target: y, weight: parseFloat(graph[x][y])};
                    linkData[i++] = link;
                }
            }

            links = linkData;

            console.log(linkData);

            var community = jLouvain()
                .nodes(nodeData)
                .edges(linkData);
            var result = community();
            console.log(result);
            console.log(nodes);

            nodes.forEach(function (node) {
                node.r = defaultRadius;
                node.cluster = result[node.id]
            });

            nodes.forEach(function (node) {
                const radius = node.r;
                const clusterID = node.cluster;
                if (!clusters[clusterID] || (radius > clusters[clusterID].r)) {
                    clusters[clusterID] = node;
                }
            });


            // var child = document.getElementsByTagName("svg");
            // if (child.length > 0) {
            //     child[0].parentNode.removeChild(child[0])
            // }

            var clu_svg = d3.select("#cluster");
            clu_svg.remove();

            const svg = d3.select('svg')
                .append('svg')
                .attr('id', "cluster")
                .attr('x', 0)
                .attr('y', 420)
                .attr('height', height_c)
                .attr('width', width_c)
                .append('g')
                .attr('transform', 'translate(' + width_c / 2 + ',' + height_c * 3 / 10 + ')');

            var link = svg.selectAll('line')
                .data(links)
                .enter().append('line');

            link
                .attr('class', 'link')
                .style('stroke', 'darkgray')
                .style('stroke-width', '2px');

            const circles = svg.append('g')
                .datum(nodes)
                .selectAll('.circle')
                .data(function (d) {
                    return d;
                })
                .enter().append('circle')
                .attr('id', function(d) {
                    return d.id;
                })
                .attr('r', function (d) {
                    return d.r;
                })
                .attr('cluster', function (d) {
                    return d.cluster;
                })
                .attr('fill', function (d) {
                    return z(d.cluster);
                })
                .attr('stroke', 'black')
                .attr('stroke-width', 1)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended)
                )
                .on("click", function() {
                    var circle =  d3.select(this);
                    var cluster = circle.attr("cluster");
                    var rect_list = new Array()
                    
                    var circles = d3.selectAll("circle").each(function (node) {
                        if (node.cluster == cluster) {
                            rect_list.push(node.id);
                        }
                    });

                    console.log(cluster);
                    console.log(rect_list);

                    rect_list.forEach(function(d){
                        rect = d3.select("body").select("svg").select("#g" + layer).select("#r" + d)
                        .attr("fill", circle.attr("fill"));
                    });
                });

            const simulation = d3.forceSimulation()
                .nodes(nodes)
                // .force('link', d3.forceLink().id(function (d) {
                //     return d.id;
                // }))
                .velocityDecay(1)
                .force('x', d3.forceX().strength(0.7))
                .force('y', d3.forceY().strength(0.7))
                .force('collide', collide)
                .force('cluster', clustering)
                .on('tick', ticked);

            // simulation.force('link')
            //     .links(links)

            function clustering(alpha) {
                nodes.forEach(function (d) {
                    const cluster = clusters[d.cluster];
                    if (cluster === d) return;
                    x = d.x - cluster.x;
                    y = d.y - cluster.y;
                    l = Math.sqrt((x * x) + (y * y));
                    const r = d.r + cluster.r;
                    if (l !== r) {
                        l = ((l - r) / l) * alpha;
                        d.x -= x *= l;
                        d.y -= y *= l;
                        cluster.x += x;
                        cluster.y += y;
                    }
                });
            }

            function ticked() {
                link.attr('x1', function (d) {
                    return d.source.x;
                })
                    .attr('y1', function (d) {
                        return d.source.y;
                    })
                    .attr('x2', function (d) {
                        return d.target.x;
                    })
                    .attr('y2', function (d) {
                        return d.target.y;
                    });

                circles.attr('cx', function (d) {
                    return d.x;
                })
                    .attr('cy', function (d) {
                        return d.y;
                    });
            }

            function collide(alpha) {
                const quadtree = d3.quadtree().x(function (d) {
                    return d.x;
                })
                    .y(function (d) {
                        return d.y;
                    })
                    .addAll(nodes);

                nodes.forEach(function (d) {
                    const r = d.r + maxRadius + Math.max(padding, clusterPadding);
                    const nx1 = d.x - r;
                    const nx2 = d.x + r;
                    const ny1 = d.y - r;
                    const ny2 = d.y + r;
                    quadtree.visit(function(quad, x1, y1, x2, y2) {
                        if(quad.data && (quad.data !== d)){
                            x = d.x - quad.data.x;
                            y = d.y - quad.data.y;
                            l = Math.sqrt((x * x) + (y * y));
                            const r = d.r + quad.data.r + (d.cluster === quad.data.cluster ? padding : clusterPadding);
                            if (l < r) {
                                l = ((l - r) / l) * alpha;
                                d.x -= x *= l;
                                d.y -= y *= l;
                                quad.data.x += x;
                                quad.data.y += y;
                            }
                        }
                        return x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;
                    });
                });
            }

            function dragstarted(d) {
                if (!d3.event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(d) {
                d.fx = d3.event.x;
                d.fy = d3.event.y;
            }

            function dragended(d) {
                if (!d3.event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
    });
}

function transition(layer) {
    d3.json("http://127.0.0.1:5000/getCorrData?l=" + layer, function(error, graph) {
        var nodes_t = new Array();
        var nodeData_t = new Array();
        var linkData_t = new Array();
        var filters_num_t = Object.keys(graph).length;
        for (var i = 0; i < filters_num_t; i++) {
            nodeData_t[i] = i;
            node_t = {id: i};
            nodes_t[i] = node_t;
        }

        var i = 0;
        for (var x = 0; x < filters_num_t; x++) {
            for (var y = x + 1; y < filters_num_t; y++){
                link_t = {source: x, target: y, weight: parseFloat(graph[x][y])};
                linkData_t[i++] = link_t;
            }
        }

        links_t = linkData_t;

        var community = jLouvain()
                .nodes(nodeData_t)
                .edges(linkData_t);
        var result_t = community();

        console.log(result_t)

        var flag = 0;
        if (Object.keys(result_t).length !== Object.keys(last_result).length) {
            flag = 3;
        } else {
            for (var id in result_t) {
                if (result_t[id] !== last_result[id]) {
                    flag += 1;
                }
            }
        }
        last_result = result_t;

        if (flag > 2) {
            nodes_t.forEach(function (node_t) {
            node_t.r = defaultRadius;
            node_t.cluster = result_t[node_t.id]
            });

            for (var i = 0; i < filters_num_t; i++) {
                clusters[i] = nodes_t[0];
            }
            nodes_t.forEach(function (node_t) {
                const radius = node_t.r;
                const clusterID = node_t.cluster;
                clusters[clusterID] = node_t;
            });

            var link_t = d3.selectAll('line')
                .data(links_t)
                .enter().append('line');

            var circles_t = d3.selectAll('circle')
                .data(nodes_t)
                .attr('r', function (d) {
                    return d.r;
                })
                .attr('cluster', function (d) {
                    return d.cluster;
                })
                .attr('fill', function (d) {
                    return z(d.cluster);
                })
                .attr('stroke', 'black')
                .attr('stroke-width', 1)
                .call(d3.drag()
                    .on("start", louvain_cluster.dragstarted)
                    .on("drag", louvain_cluster.dragged)
                    .on("end", louvain_cluster.dragended)
                );

            console.log(circles_t);

            simulation = d3.forceSimulation()
                    .nodes(nodes_t)
                    // .force('link', d3.forceLink().id(function (d) {
                    //     return d.id;
                    // }))
                    .velocityDecay(1)
                    .force('x', d3.forceX().strength(0.7))
                    .force('y', d3.forceY().strength(0.7))
                    .force('collide', collide_t)
                    .force('cluster', clustering_t)
                    .on('tick', ticked_t);

            // simulation.force('link')
            //         .links(links_t);
        }

        function clustering_t(alpha) {
            nodes_t.forEach(function (d) {
                const cluster = clusters[d.cluster];
                if (cluster === d) return;
                x = d.x - cluster.x;
                y = d.y - cluster.y;
                l = Math.sqrt((x * x) + (y * y));
                const r = d.r + cluster.r;
                if (l !== r) {
                    l = ((l - r) / l) * alpha;
                    d.x -= x *= l;
                    d.y -= y *= l;
                    cluster.x += x;
                    cluster.y += y;
                }
            });
        }

        function ticked_t() {
            link_t.attr('x1', function (d) {
                return d.source.x;
            })
                .attr('y1', function (d) {
                    return d.source.y;
                })
                .attr('x2', function (d) {
                    return d.target.x;
                })
                .attr('y2', function (d) {
                    return d.target.y;
                });

            circles_t.attr('cx', function (d) {
                return d.x;
            })
                .attr('cy', function (d) {
                    return d.y;
                });
        }

        function collide_t(alpha) {
            const quadtree = d3.quadtree().x(function (d) {
                return d.x;
            })
                .y(function (d) {
                    return d.y;
                })
                .addAll(nodes_t);

            nodes_t.forEach(function (d) {
                const r = d.r + maxRadius + Math.max(padding, clusterPadding);
                const nx1 = d.x - r;
                const nx2 = d.x + r;
                const ny1 = d.y - r;
                const ny2 = d.y + r;
                quadtree.visit(function(quad, x1, y1, x2, y2) {
                    if(quad.data && (quad.data !== d)){
                        x = d.x - quad.data.x;
                        y = d.y - quad.data.y;
                        l = Math.sqrt((x * x) + (y * y));
                        const r = d.r + quad.data.r + (d.cluster === quad.data.cluster ? padding : clusterPadding);
                        if (l < r) {
                            l = ((l - r) / l) * alpha;
                            d.x -= x *= l;
                            d.y -= y *= l;
                            quad.data.x += x;
                            quad.data.y += y;
                        }
                    }
                    return x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;
                });
            });
        }

    });
}

