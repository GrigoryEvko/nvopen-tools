// Function: sub_295A930
// Address: 0x295a930
//
void __fastcall sub_295A930(char *src, char *a2, __int64 a3)
{
  __int64 v4; // rcx
  char *v5; // r15
  __int64 v6; // rbx

  if ( a2 - src <= 112 )
  {
    sub_2958540(src, a2, a3);
  }
  else
  {
    v4 = (a2 - src) >> 4;
    v5 = &src[8 * v4];
    v6 = (8 * v4) >> 3;
    sub_295A930(src);
    sub_295A930(v5);
    sub_295A6B0(src, v5, (__int64)a2, v6, (a2 - v5) >> 3, a3);
  }
}
