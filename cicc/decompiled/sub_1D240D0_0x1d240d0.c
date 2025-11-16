// Function: sub_1D240D0
// Address: 0x1d240d0
//
__int64 __fastcall sub_1D240D0(__int64 a1, __int64 a2, __int64 a3, int a4, char a5, __int64 *a6, int a7)
{
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r12
  __int64 v15; // rsi
  _QWORD v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v11 = *a6;
  v17[0] = v11;
  if ( v11 )
    sub_1623A60((__int64)v17, v11, 2);
  v12 = sub_145CBF0(*(__int64 **)(a1 + 648), 56, 16);
  v13 = v17[0];
  *(_QWORD *)(v12 + 16) = a2;
  v14 = v12;
  *(_QWORD *)(v12 + 24) = a3;
  *(_QWORD *)(v12 + 32) = v13;
  if ( v13 )
  {
    sub_1623A60(v12 + 32, v13, 2);
    v15 = v17[0];
    *(_DWORD *)v14 = a4;
    *(_BYTE *)(v14 + 48) = a5;
    *(_DWORD *)(v14 + 40) = a7;
    *(_BYTE *)(v14 + 49) = 0;
    *(_DWORD *)(v14 + 44) = 2;
    if ( v15 )
      sub_161E7C0((__int64)v17, v15);
  }
  else
  {
    *(_BYTE *)(v12 + 48) = a5;
    *(_BYTE *)(v12 + 49) = 0;
    *(_DWORD *)(v12 + 40) = a7;
    *(_DWORD *)(v12 + 44) = 2;
    *(_DWORD *)v12 = a4;
  }
  return v14;
}
