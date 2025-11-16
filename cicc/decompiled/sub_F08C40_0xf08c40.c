// Function: sub_F08C40
// Address: 0xf08c40
//
__int64 __fastcall sub_F08C40(__int64 *src, __int64 *a2, char *a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (a2 - src + 1) / 2;
  v10 = 8 * v6;
  v7 = &src[v6];
  if ( v6 <= a4 )
  {
    sub_F07460(src, &src[v6], a3);
    sub_F07460(v7, a2, a3);
  }
  else
  {
    sub_F08C40(src);
    sub_F08C40(v7);
  }
  sub_F08760(src, v7, (__int64)a2, v10 >> 3, a2 - v7, (void **)a3, a4);
  return v9;
}
