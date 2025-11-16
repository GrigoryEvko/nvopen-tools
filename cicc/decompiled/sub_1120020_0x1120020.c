// Function: sub_1120020
// Address: 0x1120020
//
bool __fastcall sub_1120020(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rdx
  __int64 v3; // rsi
  _BYTE *v4; // r12
  unsigned int v5; // r13d
  __int64 v6; // rax
  __int64 v8; // rdx
  _BYTE *v9; // rax

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  **(_QWORD **)a1 = a2;
  if ( (unsigned __int8)(*(_BYTE *)a2 - 55) <= 1u )
  {
    v2 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
       ? *(_QWORD **)(a2 - 8)
       : (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *v2 )
    {
      **(_QWORD **)(a1 + 8) = *v2;
      v3 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v4 = *(_BYTE **)(v3 + 32);
      if ( *v4 == 17
        || (v8 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v4 + 1) + 8LL) - 17, (unsigned int)v8 <= 1)
        && *v4 <= 0x15u
        && (v9 = sub_AD7630(*(_QWORD *)(v3 + 32), 1, v8), (v4 = v9) != 0)
        && *v9 == 17 )
      {
        v5 = *((_DWORD *)v4 + 8);
        if ( v5 <= 0x40 )
        {
          v6 = *((_QWORD *)v4 + 3);
        }
        else
        {
          if ( v5 - (unsigned int)sub_C444A0((__int64)(v4 + 24)) > 0x40 )
            return 0;
          v6 = **((_QWORD **)v4 + 3);
        }
        return *(_QWORD *)(a1 + 16) == v6;
      }
    }
  }
  return 0;
}
