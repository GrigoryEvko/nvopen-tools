// Function: sub_2529340
// Address: 0x2529340
//
__int64 __fastcall sub_2529340(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rcx
  void (__fastcall *v11)(_BYTE *, __int64, __int64); // rax
  unsigned int v12; // r12d
  __int64 v14; // [rsp+8h] [rbp-68h]
  _BYTE v15[16]; // [rsp+20h] [rbp-50h] BYREF
  void (__fastcall *v16)(_BYTE *, _BYTE *, __int64); // [rsp+30h] [rbp-40h]
  __int64 v17; // [rsp+38h] [rbp-38h]

  v16 = 0;
  v10 = sub_B43CB0(a3);
  v11 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a6 + 16);
  if ( v11 )
  {
    v14 = v10;
    v11(v15, a6, 2);
    v10 = v14;
    v17 = *(_QWORD *)(a6 + 24);
    v16 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a6 + 16);
  }
  v12 = sub_2528D80(a1, a2, a3, v10, a4, a5, (__int64)v15);
  if ( v16 )
    v16(v15, v15, 3);
  return v12;
}
