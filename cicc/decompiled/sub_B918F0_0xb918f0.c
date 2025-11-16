// Function: sub_B918F0
// Address: 0xb918f0
//
__int64 __fastcall sub_B918F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 v3; // dl
  __int64 v4; // rdx
  __int64 v5; // rcx
  unsigned __int8 v6; // dl
  __int64 v7; // rdx
  __int64 v8; // rsi
  _QWORD *v9; // rdx
  _QWORD *v10; // rcx

  result = a2;
  if ( !a1 || !a2 )
    return 0;
  v3 = *(_BYTE *)(a1 - 16);
  if ( (v3 & 2) != 0 )
    v4 = *(_QWORD *)(a1 - 32);
  else
    v4 = a1 - 8LL * ((v3 >> 2) & 0xF) - 16;
  v5 = *(_QWORD *)(*(_QWORD *)v4 + 136LL);
  v6 = *(_BYTE *)(a2 - 16);
  if ( (v6 & 2) != 0 )
    v7 = *(_QWORD *)(a2 - 32);
  else
    v7 = a2 - 8LL * ((v6 >> 2) & 0xF) - 16;
  v8 = *(_QWORD *)(*(_QWORD *)v7 + 136LL);
  v9 = *(_QWORD **)(v5 + 24);
  if ( *(_DWORD *)(v5 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v10 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  if ( v10 > v9 )
    return a1;
  return result;
}
