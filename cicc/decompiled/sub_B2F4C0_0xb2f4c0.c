// Function: sub_B2F4C0
// Address: 0xb2f4c0
//
__int64 __fastcall sub_B2F4C0(__int64 a1, __int64 a2, int a3, _BYTE *a4)
{
  __int64 v6; // rax
  __int64 v8; // [rsp+8h] [rbp-58h] BYREF
  _BYTE v9[8]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v10; // [rsp+18h] [rbp-48h]
  int v11; // [rsp+20h] [rbp-40h]
  unsigned int v12; // [rsp+28h] [rbp-38h]

  sub_B2F150((__int64)v9, a1);
  if ( !a4 && v11 )
    a4 = v9;
  v8 = sub_B2BE50(a1);
  v6 = sub_B8C360(&v8, a2, a3 == 1, a4);
  sub_B99110(a1, 2, v6);
  return sub_C7D6A0(v10, 8LL * v12, 8);
}
