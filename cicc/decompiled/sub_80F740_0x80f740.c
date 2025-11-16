// Function: sub_80F740
// Address: 0x80f740
//
unsigned __int8 *__fastcall sub_80F740(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD v7[4]; // [rsp+0h] [rbp-60h] BYREF
  char v8; // [rsp+20h] [rbp-40h]
  __int64 v9; // [rsp+28h] [rbp-38h]
  __int64 v10; // [rsp+30h] [rbp-30h]
  int v11; // [rsp+38h] [rbp-28h]
  char v12; // [rsp+3Ch] [rbp-24h]
  __int64 v13; // [rsp+40h] [rbp-20h]

  v7[3] = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  sub_809110(a1, a2, a3, a4, a5, a6, 0, 0, 0);
  sub_823800(qword_4F18BE0);
  sub_80F5E0(a1, 0, v7);
  return sub_80B290(0, 1, (__int64)v7);
}
