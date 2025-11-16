// Function: sub_13D6F20
// Address: 0x13d6f20
//
_BOOL8 __fastcall sub_13D6F20(_QWORD **a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rcx
  int v5; // edx
  _BOOL4 v6; // r12d
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  _BYTE *v10; // rdi
  unsigned __int8 v11; // al
  unsigned __int64 v13; // rax
  void *v14; // rcx
  __int64 v15; // rax

  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 <= 0x17u )
  {
    if ( (_BYTE)v2 == 5 )
    {
      v13 = *(unsigned __int16 *)(a2 + 18);
      if ( (unsigned __int16)v13 <= 0x17u )
      {
        v14 = &loc_80A800;
        v5 = (unsigned __int16)v13;
        if ( _bittest64((const __int64 *)&v14, v13) )
          goto LABEL_4;
      }
    }
    return 0;
  }
  if ( (unsigned __int8)v2 > 0x2Fu )
    return 0;
  v4 = 0x80A800000000LL;
  v5 = (unsigned __int8)v2 - 24;
  if ( !_bittest64(&v4, v2) )
    return 0;
LABEL_4:
  if ( v5 != 23 )
    return 0;
  v6 = (*(_BYTE *)(a2 + 17) & 2) != 0;
  if ( (*(_BYTE *)(a2 + 17) & 2) == 0 )
    return 0;
  v7 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
     ? *(__int64 **)(a2 - 8)
     : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v8 = *v7;
  if ( !v8 )
    return 0;
  **a1 = v8;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v9 = *(_QWORD *)(a2 - 8);
  else
    v9 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v10 = *(_BYTE **)(v9 + 24);
  v11 = v10[16];
  if ( v11 != 13 )
  {
    LOBYTE(v6) = v11 <= 0x10u && *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16;
    if ( v6 )
    {
      v15 = sub_15A1020(v10);
      if ( v15 )
      {
        if ( *(_BYTE *)(v15 + 16) == 13 )
        {
          *a1[1] = v15 + 24;
          return v6;
        }
      }
    }
    return 0;
  }
  *a1[1] = v10 + 24;
  return v6;
}
