// Function: sub_10E4520
// Address: 0x10e4520
//
_BOOL8 __fastcall sub_10E4520(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int64 v3; // rax
  _BOOL4 v4; // r12d
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int16 v11; // dx
  __int64 v12; // rdx
  _BYTE *v13; // rax
  __int64 v14; // rdx
  _BYTE *v15; // rax

  v3 = *a2;
  if ( (unsigned __int8)v3 <= 0x1Cu )
  {
    if ( (_BYTE)v3 != 5 )
      return 0;
    v11 = *((_WORD *)a2 + 1);
    if ( (v11 & 0xFFFD) != 0xD && (v11 & 0xFFF7) != 0x11 )
      return 0;
    if ( v11 != 13 )
      return 0;
  }
  else
  {
    if ( (unsigned __int8)v3 > 0x36u )
      goto LABEL_3;
    v8 = 0x40540000000000LL;
    if ( !_bittest64(&v8, v3) || (_BYTE)v3 != 42 )
      return 0;
  }
  if ( (a2[1] & 4) == 0 )
    return 0;
  v9 = *((_QWORD *)a2 - 8);
  if ( v9 )
  {
    **(_QWORD **)a1 = v9;
    v10 = *((_QWORD *)a2 - 4);
    if ( *(_BYTE *)v10 == 17 )
    {
      v4 = 1;
      **(_QWORD **)(a1 + 8) = v10 + 24;
      return v4;
    }
    v12 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17;
    if ( (unsigned int)v12 <= 1 && *(_BYTE *)v10 <= 0x15u )
    {
      v13 = sub_AD7630(v10, *(unsigned __int8 *)(a1 + 16), v12);
      if ( v13 )
      {
        if ( *v13 == 17 )
        {
          v4 = 1;
          **(_QWORD **)(a1 + 8) = v13 + 24;
          return v4;
        }
      }
    }
    LOBYTE(v3) = *a2;
  }
LABEL_3:
  if ( (_BYTE)v3 != 58 )
    return 0;
  v4 = (a2[1] & 2) != 0;
  if ( (a2[1] & 2) == 0 )
    return 0;
  v5 = *((_QWORD *)a2 - 8);
  if ( !v5 )
    return 0;
  **(_QWORD **)(a1 + 24) = v5;
  v6 = *((_QWORD *)a2 - 4);
  if ( *(_BYTE *)v6 != 17 )
  {
    v14 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17;
    if ( (unsigned int)v14 <= 1 && *(_BYTE *)v6 <= 0x15u )
    {
      v15 = sub_AD7630(v6, *(unsigned __int8 *)(a1 + 40), v14);
      if ( v15 )
      {
        if ( *v15 == 17 )
        {
          **(_QWORD **)(a1 + 32) = v15 + 24;
          return v4;
        }
      }
    }
    return 0;
  }
  **(_QWORD **)(a1 + 32) = v6 + 24;
  return v4;
}
