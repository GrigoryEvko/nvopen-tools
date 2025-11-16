// Function: sub_1A592E0
// Address: 0x1a592e0
//
__int64 __fastcall sub_1A592E0(char *src, char *a2, char *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 *v9; // r10
  __int64 v10; // r9
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  char *srca; // [rsp+8h] [rbp-38h]

  v8 = (((a2 - src) >> 3) + 1) / 2;
  v13 = 8 * v8;
  srca = &src[8 * v8];
  if ( v8 <= a4 )
  {
    sub_1A50C10(src, &src[8 * v8], a3, a5);
    sub_1A50C10(srca, a2, a3, a5);
    v10 = v13;
    v9 = (__int64 *)srca;
  }
  else
  {
    sub_1A592E0(src);
    sub_1A592E0(srca);
    v9 = (__int64 *)srca;
    v10 = v13;
  }
  sub_1A58B80((__int64 *)src, v9, (__int64)a2, v10 >> 3, (a2 - (char *)v9) >> 3, a3, a4, a5);
  return v12;
}
