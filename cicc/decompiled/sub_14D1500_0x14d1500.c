// Function: sub_14D1500
// Address: 0x14d1500
//
__int64 __fastcall sub_14D1500(__int64 a1, char a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // r15d
  unsigned int v7; // r15d
  __int64 v8; // rax
  __int64 v9; // rdi
  int v10; // eax
  char v12; // [rsp+17h] [rbp-39h] BYREF
  _QWORD v13[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(_DWORD *)(a3 + 8);
  v12 = 0;
  v7 = v6 >> 8;
  v8 = sub_16982C0(a1, v13, a3, a4);
  v9 = a1 + 8;
  if ( *(_QWORD *)(a1 + 8) == v8 )
    v10 = sub_169E030(v9, (unsigned int)v13, 1, v7, 1, a2 != 0 ? 3 : 0, (__int64)&v12);
  else
    v10 = sub_169A0A0(v9, v13, 1, v7, 1, a2 != 0 ? 3 : 0, &v12);
  if ( !v10 || a2 == 1 && v10 == 16 )
    return sub_15A0680(a3, v13[0], 1);
  else
    return 0;
}
