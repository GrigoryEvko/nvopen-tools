// Function: sub_F15770
// Address: 0xf15770
//
bool __fastcall sub_F15770(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  _BYTE *v4; // r12
  unsigned int v5; // r13d
  __int64 v6; // rax
  __int64 v8; // rdx
  _BYTE *v9; // rax

  if ( (unsigned __int8)(*(_BYTE *)a2 - 55) > 1u )
    return 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v2 = **(_QWORD **)(a2 - 8);
    if ( !v2 )
      return 0;
  }
  else
  {
    v2 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( !v2 )
      return 0;
  }
  **(_QWORD **)a1 = v2;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(a2 - 8);
  else
    v3 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v4 = *(_BYTE **)(v3 + 32);
  if ( *v4 == 17 )
  {
LABEL_7:
    v5 = *((_DWORD *)v4 + 8);
    if ( v5 <= 0x40 )
    {
      v6 = *((_QWORD *)v4 + 3);
      return *(_QWORD *)(a1 + 8) == v6;
    }
    if ( v5 - (unsigned int)sub_C444A0((__int64)(v4 + 24)) <= 0x40 )
    {
      v6 = **((_QWORD **)v4 + 3);
      return *(_QWORD *)(a1 + 8) == v6;
    }
    return 0;
  }
  v8 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v4 + 1) + 8LL) - 17;
  if ( (unsigned int)v8 <= 1 && *v4 <= 0x15u )
  {
    v9 = sub_AD7630(*(_QWORD *)(v3 + 32), 0, v8);
    v4 = v9;
    if ( v9 )
    {
      if ( *v9 != 17 )
        return 0;
      goto LABEL_7;
    }
  }
  return 0;
}
