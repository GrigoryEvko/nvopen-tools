// Function: sub_1F21DF0
// Address: 0x1f21df0
//
void __fastcall sub_1F21DF0(char *src, char *a2, __int64 a3)
{
  __int64 v4; // rcx
  char *v5; // r15
  __int64 v6; // rbx

  if ( a2 - src <= 56 )
  {
    sub_1F20C70(src, a2, a3);
  }
  else
  {
    v4 = (a2 - src) >> 3;
    v5 = &src[4 * v4];
    v6 = (4 * v4) >> 2;
    sub_1F21DF0(src);
    sub_1F21DF0(v5);
    sub_1F21C50(src, v5, (__int64)a2, v6, (a2 - v5) >> 2, a3);
  }
}
