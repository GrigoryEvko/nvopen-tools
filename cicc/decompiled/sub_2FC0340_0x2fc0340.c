// Function: sub_2FC0340
// Address: 0x2fc0340
//
void __fastcall sub_2FC0340(char *src, char *a2, __int64 a3)
{
  __int64 v4; // rcx
  char *v5; // r15
  __int64 v6; // rbx

  if ( a2 - src <= 56 )
  {
    sub_2FBF180(src, a2, a3);
  }
  else
  {
    v4 = (a2 - src) >> 3;
    v5 = &src[4 * v4];
    v6 = (4 * v4) >> 2;
    sub_2FC0340(src);
    sub_2FC0340(v5);
    sub_2FC01B0(src, v5, (__int64)a2, v6, (a2 - v5) >> 2, a3);
  }
}
