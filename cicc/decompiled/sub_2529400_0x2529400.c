// Function: sub_2529400
// Address: 0x2529400
//
__int64 __fastcall sub_2529400(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  void (__fastcall *v7)(_BYTE *, __int64, __int64); // rax
  unsigned int v9; // r13d
  __int64 v11; // [rsp+0h] [rbp-60h]
  __int64 v12; // [rsp+8h] [rbp-58h]
  _BYTE v13[16]; // [rsp+10h] [rbp-50h] BYREF
  void (__fastcall *v14)(_BYTE *, _BYTE *, __int64); // [rsp+20h] [rbp-40h]
  __int64 v15; // [rsp+28h] [rbp-38h]

  v7 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a6 + 16);
  v14 = 0;
  if ( v7 )
  {
    v11 = a5;
    v12 = a4;
    v7(v13, a6, 2);
    a5 = v11;
    a4 = v12;
    v15 = *(_QWORD *)(a6 + 24);
    v14 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a6 + 16);
  }
  v9 = sub_2528D80(a1, a2, 0, a3, a4, a5, (__int64)v13);
  if ( v14 )
    v14(v13, v13, 3);
  return v9;
}
