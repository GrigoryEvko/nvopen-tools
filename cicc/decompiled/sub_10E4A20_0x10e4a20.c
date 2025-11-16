// Function: sub_10E4A20
// Address: 0x10e4a20
//
__int64 __fastcall sub_10E4A20(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) || *(_BYTE *)a2 != 86 )
    return 0;
  v4 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
     ? *(_QWORD **)(a2 - 8)
     : (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( !*v4 )
    return 0;
  **a1 = *v4;
  v5 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v6 = *(_QWORD *)(v5 + 32);
  if ( !v6 )
    return 0;
  *a1[1] = v6;
  v7 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v8 = *(_QWORD *)(v7 + 64);
  if ( !v8 )
    return 0;
  *a1[2] = v8;
  return 1;
}
