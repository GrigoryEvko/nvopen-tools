// Function: sub_29BF9B0
// Address: 0x29bf9b0
//
void __fastcall sub_29BF9B0(char *src, char *a2, __int64 *a3)
{
  __int64 v4; // rcx
  char *v5; // r15
  __int64 v6; // rbx

  if ( a2 - src <= 112 )
  {
    sub_29BF2A0(src, a2, a3);
  }
  else
  {
    v4 = (a2 - src) >> 4;
    v5 = &src[8 * v4];
    v6 = (8 * v4) >> 3;
    sub_29BF9B0(src);
    sub_29BF9B0(v5);
    sub_29BF830(src, v5, (__int64)a2, v6, (a2 - v5) >> 3, a3);
  }
}
