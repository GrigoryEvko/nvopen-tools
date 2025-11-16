// Function: sub_F7A870
// Address: 0xf7a870
//
void __fastcall sub_F7A870(char *src, char *a2)
{
  __int64 v2; // rcx
  char *v3; // r14
  __int64 v4; // rbx

  if ( a2 - src <= 112 )
  {
    sub_F7A2A0(src, a2);
  }
  else
  {
    v2 = (a2 - src) >> 4;
    v3 = &src[8 * v2];
    v4 = (8 * v2) >> 3;
    sub_F7A870(src);
    sub_F7A870(v3);
    sub_F7A6D0((__int64)src, v3, (__int64)a2, v4, (a2 - v3) >> 3);
  }
}
