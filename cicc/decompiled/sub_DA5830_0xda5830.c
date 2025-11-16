// Function: sub_DA5830
// Address: 0xda5830
//
__int64 __fastcall sub_DA5830(unsigned __int64 *src, unsigned __int64 *a2, char *a3, void *a4, _QWORD **a5)
{
  __int64 v8; // rax
  unsigned __int64 *v9; // r10
  __int64 v10; // r9
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  unsigned __int64 *srca; // [rsp+8h] [rbp-38h]

  v8 = (a2 - src + 1) / 2;
  v13 = 8 * v8;
  srca = &src[v8];
  if ( v8 <= (__int64)a4 )
  {
    sub_DA5250(src, &src[v8], a3, a5);
    sub_DA5250(srca, a2, a3, a5);
    v10 = v13;
    v9 = srca;
  }
  else
  {
    sub_DA5830(src);
    sub_DA5830(srca);
    v9 = srca;
    v10 = v13;
  }
  sub_DA5350(src, v9, (__int64)a2, v10 >> 3, a2 - v9, (unsigned __int64 *)a3, a4, a5);
  return v12;
}
