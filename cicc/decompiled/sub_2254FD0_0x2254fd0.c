// Function: sub_2254FD0
// Address: 0x2254fd0
//
__int64 sub_2254FD0()
{
  int v0; // eax

  v0 = __wcscoll_l();
  return (v0 >> 30) | (unsigned int)(v0 != 0);
}
