// Function: sub_2254F80
// Address: 0x2254f80
//
__int64 sub_2254F80()
{
  int v0; // eax

  v0 = __strcoll_l();
  return (v0 >> 30) | (unsigned int)(v0 != 0);
}
