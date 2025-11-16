// Function: sub_1F225E0
// Address: 0x1f225e0
//
__int64 __fastcall sub_1F225E0(char *src, char *a2, char *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  char *v9; // r10
  __int64 v10; // r9
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  char *srca; // [rsp+8h] [rbp-38h]

  v8 = (((a2 - src) >> 2) + 1) / 2;
  v13 = 4 * v8;
  srca = &src[4 * v8];
  if ( v8 <= a4 )
  {
    sub_1F20D80(src, &src[4 * v8], a3, a5);
    sub_1F20D80(srca, a2, a3, a5);
    v10 = v13;
    v9 = srca;
  }
  else
  {
    sub_1F225E0(src);
    sub_1F225E0(srca);
    v9 = srca;
    v10 = v13;
  }
  sub_1F22140(src, v9, (__int64)a2, v10 >> 2, (a2 - v9) >> 2, (unsigned int *)a3, a4, a5);
  return v12;
}
