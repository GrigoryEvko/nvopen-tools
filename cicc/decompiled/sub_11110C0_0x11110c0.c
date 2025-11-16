// Function: sub_11110C0
// Address: 0x11110c0
//
__int64 __fastcall sub_11110C0(_QWORD **a1, __int64 a2)
{
  _QWORD *v2; // rdx
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rax

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v2 = *(_QWORD **)(a2 - 8);
  else
    v2 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( !*v2 )
    return 0;
  **a1 = *v2;
  v3 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v4 = *(_QWORD *)(v3 + 32);
  if ( !v4 )
    return 0;
  *a1[1] = v4;
  v5 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v6 = *(_QWORD *)(v5 + 64);
  if ( !v6 )
    return 0;
  *a1[2] = v6;
  return 1;
}
