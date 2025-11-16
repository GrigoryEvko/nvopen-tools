// Function: sub_310A860
// Address: 0x310a860
//
__int64 __fastcall sub_310A860(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax

  *a1 = a2;
  a1[1] = a4;
  v6 = sub_D95540(a4);
  v7 = sub_DA2C50(a2, v6, 0, 0);
  v8 = *a1;
  a1[4] = (__int64)v7;
  v9 = sub_D95540(a4);
  a1[5] = (__int64)sub_DA2C50(v8, v9, 1, 0);
  return sub_310A840(a1, a3);
}
