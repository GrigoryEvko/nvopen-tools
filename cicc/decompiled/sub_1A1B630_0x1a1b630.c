// Function: sub_1A1B630
// Address: 0x1a1b630
//
__int64 __fastcall sub_1A1B630(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rsi
  unsigned __int8 *v5; // rsi
  _QWORD v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_16498A0(a2);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 24) = v2;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 40);
  result = a2 + 24;
  *(_QWORD *)(a1 + 16) = a2 + 24;
  v4 = *(_QWORD *)(a2 + 48);
  v6[0] = v4;
  if ( v4 )
  {
    result = sub_1623A60((__int64)v6, v4, 2);
    if ( *(_QWORD *)a1 )
      result = sub_161E7C0(a1, *(_QWORD *)a1);
    v5 = (unsigned __int8 *)v6[0];
    *(_QWORD *)a1 = v6[0];
    if ( v5 )
      return sub_1623210((__int64)v6, v5, a1);
  }
  return result;
}
