// Function: sub_28EC030
// Address: 0x28ec030
//
__int64 __fastcall sub_28EC030(char *src, char *a2, char *a3, char *a4)
{
  __int64 v6; // rax
  char *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (((a2 - src) >> 3) + 1) / 2;
  v10 = 8 * v6;
  v7 = &src[8 * v6];
  if ( v6 <= (__int64)a4 )
  {
    sub_28EA080(src, &src[8 * v6], a3);
    sub_28EA080(v7, a2, a3);
  }
  else
  {
    sub_28EC030(src);
    sub_28EC030(v7);
  }
  sub_28EBBC0(src, v7, (__int64)a2, v10 >> 3, (a2 - v7) >> 3, a3, a4);
  return v9;
}
