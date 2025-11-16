// Function: sub_25F8FC0
// Address: 0x25f8fc0
//
void __fastcall sub_25F8FC0(char *src, char *a2)
{
  __int64 v2; // rcx
  char *v3; // r14
  __int64 v4; // rbx

  if ( a2 - src <= 112 )
  {
    sub_25F8290(src, a2);
  }
  else
  {
    v2 = (a2 - src) >> 4;
    v3 = &src[8 * v2];
    v4 = (8 * v2) >> 3;
    sub_25F8FC0(src);
    sub_25F8FC0(v3);
    sub_25F8DC0((__int64 *)src, (__int64 *)v3, (__int64)a2, v4, (a2 - v3) >> 3);
  }
}
