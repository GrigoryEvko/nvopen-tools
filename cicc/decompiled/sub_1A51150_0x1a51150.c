// Function: sub_1A51150
// Address: 0x1a51150
//
void __fastcall sub_1A51150(char *src, char *a2, __int64 a3)
{
  __int64 v4; // rcx
  char *v5; // r15
  __int64 v6; // rbx

  if ( a2 - src <= 112 )
  {
    sub_1A508F0(src, a2, a3);
  }
  else
  {
    v4 = (a2 - src) >> 4;
    v5 = &src[8 * v4];
    v6 = (8 * v4) >> 3;
    sub_1A51150(src);
    sub_1A51150(v5);
    sub_1A50ED0(src, v5, (__int64)a2, v6, (a2 - v5) >> 3, a3);
  }
}
