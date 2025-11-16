// Function: sub_1A1B810
// Address: 0x1a1b810
//
__int64 __fastcall sub_1A1B810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned __int8 *v7; // rsi
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 result; // rax
  _QWORD v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = sub_16498A0(a2);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = v5;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  *(_QWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 16) = a2 + 24;
  v6 = *(_QWORD *)(a2 + 48);
  v11[0] = v6;
  if ( v6 )
  {
    sub_1623A60((__int64)v11, v6, 2);
    if ( *(_QWORD *)a1 )
      sub_161E7C0(a1, *(_QWORD *)a1);
    v7 = (unsigned __int8 *)v11[0];
    *(_QWORD *)a1 = v11[0];
    if ( v7 )
      sub_1623210((__int64)v11, v7, a1);
  }
  v8 = *(_QWORD **)(a1 + 24);
  *(_QWORD *)(a1 + 104) = a1 + 120;
  *(_QWORD *)(a1 + 112) = 0x400000000LL;
  v9 = sub_1643350(v8);
  result = sub_159C470(v9, 0, 0);
  *(_QWORD *)(a1 + 184) = a3;
  *(_QWORD *)(a1 + 136) = a1 + 152;
  *(_QWORD *)(a1 + 144) = 0x400000001LL;
  *(_QWORD *)(a1 + 152) = result;
  return result;
}
