// Function: sub_1990120
// Address: 0x1990120
//
__int64 __fastcall sub_1990120(__int64 *src, __int64 *a2, char *a3, __int64 a4, __int64 *a5)
{
  __int64 v8; // rax
  __int64 *v9; // r10
  __int64 v10; // r9
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  __int64 *srca; // [rsp+8h] [rbp-38h]

  v8 = (a2 - src + 1) / 2;
  v13 = 8 * v8;
  srca = &src[v8];
  if ( v8 <= a4 )
  {
    sub_198EBB0(src, &src[v8], a3, a5);
    sub_198EBB0(srca, a2, a3, a5);
    v10 = v13;
    v9 = srca;
  }
  else
  {
    sub_1990120(src);
    sub_1990120(srca);
    v9 = srca;
    v10 = v13;
  }
  sub_198FC40(src, v9, (char *)a2, v10 >> 3, a2 - v9, (void **)a3, a4, a5);
  return v12;
}
