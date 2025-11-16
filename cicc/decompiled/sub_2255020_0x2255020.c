// Function: sub_2255020
// Address: 0x2255020
//
__int64 sub_2255020()
{
  int v0; // eax

  v0 = __strcoll_l();
  return (v0 >> 30) | (unsigned int)(v0 != 0);
}
