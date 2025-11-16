// Function: sub_3009400
// Address: 0x3009400
//
__int64 __fastcall sub_3009400(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v5; // r12d
  char v6; // bl
  __int64 *v7; // rax
  _QWORD v9[3]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v10; // [rsp+18h] [rbp-18h]

  v5 = a4;
  v6 = a5;
  v9[0] = a2;
  v9[1] = a3;
  v7 = (__int64 *)sub_3007410((__int64)v9, a1, a3, a4, a5, (__int64)a1);
  LODWORD(v10) = v5;
  BYTE4(v10) = v6;
  sub_BCE1B0(v7, v10);
  return 0;
}
