// Function: sub_F15890
// Address: 0xf15890
//
__int64 __fastcall sub_F15890(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v7; // rdx
  _BYTE *v8; // rax

  if ( (unsigned __int8)(*(_BYTE *)a2 - 55) > 1u )
    return 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) == 0 )
  {
    v3 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( v3 )
      goto LABEL_4;
    return 0;
  }
  v3 = **(_QWORD **)(a2 - 8);
  if ( !v3 )
    return 0;
LABEL_4:
  **(_QWORD **)a1 = v3;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(_QWORD *)(a2 - 8);
  else
    v4 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v5 = *(_QWORD *)(v4 + 32);
  if ( *(_BYTE *)v5 == 17 )
  {
    **(_QWORD **)(a1 + 8) = v5 + 24;
    return 1;
  }
  v7 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
  if ( (unsigned int)v7 > 1 )
    return 0;
  if ( *(_BYTE *)v5 > 0x15u )
    return 0;
  v8 = sub_AD7630(v5, *(unsigned __int8 *)(a1 + 16), v7);
  if ( !v8 || *v8 != 17 )
    return 0;
  **(_QWORD **)(a1 + 8) = v8 + 24;
  return 1;
}
