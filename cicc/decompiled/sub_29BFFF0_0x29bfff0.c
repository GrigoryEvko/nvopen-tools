// Function: sub_29BFFF0
// Address: 0x29bfff0
//
__int64 __fastcall sub_29BFFF0(char *src, char *a2, char *a3, char *a4, __int64 *a5)
{
  __int64 v8; // rax
  char *v9; // r10
  __int64 v10; // r9
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  char *srca; // [rsp+8h] [rbp-38h]

  v8 = (((a2 - src) >> 3) + 1) / 2;
  v13 = 8 * v8;
  srca = &src[8 * v8];
  if ( v8 <= (__int64)a4 )
  {
    sub_29BF380(src, &src[8 * v8], a3, a5);
    sub_29BF380(srca, a2, a3, a5);
    v10 = v13;
    v9 = srca;
  }
  else
  {
    sub_29BFFF0(src);
    sub_29BFFF0(srca);
    v9 = srca;
    v10 = v13;
  }
  sub_29BFD00(src, v9, (__int64)a2, v10 >> 3, (a2 - v9) >> 3, a3, a4, a5);
  return v12;
}
