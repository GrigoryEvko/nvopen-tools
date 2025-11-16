// Function: sub_2255070
// Address: 0x2255070
//
__int64 sub_2255070()
{
  int v0; // eax

  v0 = __wcscoll_l();
  return (v0 >> 30) | (unsigned int)(v0 != 0);
}
