// Function: sub_179DD80
// Address: 0x179dd80
//
_BOOL8 __fastcall sub_179DD80(_QWORD **a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v4; // al
  _BOOL4 v6; // r12d
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rdx
  _BYTE *v11; // rdi
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rax
  unsigned __int8 v15; // al

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 <= 0x17u )
  {
    if ( v4 != 5 )
      return 0;
    v8 = *(unsigned __int16 *)(a2 + 18);
    if ( (unsigned __int16)(*(_WORD *)(a2 + 18) - 24) > 1u && (unsigned int)(v8 - 17) > 1 )
      return 0;
    v6 = (*(_BYTE *)(a2 + 17) & 2) != 0;
    if ( (*(_BYTE *)(a2 + 17) & 2) == 0 )
      return 0;
    if ( (unsigned int)(v8 - 24) > 1 )
      return 0;
    v9 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( !v9 )
      return 0;
    **a1 = v9;
    v10 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v11 = *(_BYTE **)(a2 + 24 * (1 - v10));
    if ( v11[16] == 13 )
    {
LABEL_25:
      *a1[1] = v11 + 24;
      return v6;
    }
    if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) != 16 )
      return 0;
  }
  else
  {
    a4 = (unsigned int)v4 - 41;
    if ( (unsigned int)a4 > 1 && (unsigned __int8)(v4 - 48) > 1u )
      return 0;
    v6 = (*(_BYTE *)(a2 + 17) & 2) != 0;
    if ( (*(_BYTE *)(a2 + 17) & 2) == 0 || (unsigned int)v4 - 48 > 1 )
      return 0;
    v13 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
        ? *(__int64 **)(a2 - 8)
        : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v14 = *v13;
    if ( !v14 )
      return 0;
    **a1 = v14;
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      a2 = *(_QWORD *)(a2 - 8);
    else
      a2 -= 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v11 = *(_BYTE **)(a2 + 24);
    v15 = v11[16];
    if ( v15 == 13 )
      goto LABEL_25;
    v10 = *(_QWORD *)v11;
    LOBYTE(v6) = v15 <= 0x10u && *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 16;
    if ( !v6 )
      return 0;
  }
  v12 = sub_15A1020(v11, a2, v10, a4);
  if ( v12 && *(_BYTE *)(v12 + 16) == 13 )
  {
    *a1[1] = v12 + 24;
    return v6;
  }
  return 0;
}
