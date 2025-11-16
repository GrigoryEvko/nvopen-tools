// Function: sub_14B00C0
// Address: 0x14b00c0
//
__int64 __fastcall sub_14B00C0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r12
  __int64 v4; // rax
  _BYTE *v5; // rdx
  _BYTE *v7; // r13
  _BYTE *v8; // rdi
  int v9; // ecx
  _BYTE *v10; // r12
  _BYTE *v11; // r13
  int v12; // eax
  unsigned __int8 v13; // al
  __int64 v14; // rax

  if ( *(_BYTE *)(a2 + 16) != 79 )
    goto LABEL_2;
  v4 = *(_QWORD *)(a2 - 72);
  if ( *(_BYTE *)(v4 + 16) != 76 )
    goto LABEL_2;
  v5 = *(_BYTE **)(a2 - 48);
  v2 = *(_BYTE **)(v4 - 48);
  v7 = *(_BYTE **)(v4 - 24);
  v8 = *(_BYTE **)(a2 - 24);
  if ( v5 == v2 && v8 == v7 )
  {
    v9 = *(unsigned __int16 *)(v4 + 18);
  }
  else
  {
    if ( v5 != v7 || v8 != v2 )
      goto LABEL_12;
    v9 = *(unsigned __int16 *)(v4 + 18);
    if ( v5 != v2 )
    {
      if ( (unsigned int)sub_15FF0F0(*(_WORD *)(v4 + 18) & 0x7FFF) - 4 > 1 || v2 != *(_BYTE **)a1 )
      {
LABEL_23:
        if ( *(_BYTE *)(a2 + 16) != 79 )
          goto LABEL_2;
        v4 = *(_QWORD *)(a2 - 72);
LABEL_10:
        if ( *(_BYTE *)(v4 + 16) == 76 )
        {
          v5 = *(_BYTE **)(a2 - 48);
          v8 = *(_BYTE **)(a2 - 24);
          goto LABEL_12;
        }
LABEL_2:
        LODWORD(v2) = 0;
        return (unsigned int)v2;
      }
LABEL_30:
      v13 = v7[16];
      if ( v13 == 14 )
      {
        LODWORD(v2) = 1;
        **(_QWORD **)(a1 + 8) = v7 + 24;
        return (unsigned int)v2;
      }
      LOBYTE(v2) = v13 <= 0x10u && *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16;
      if ( (_BYTE)v2 )
      {
        v14 = sub_15A1020(v7);
        if ( v14 )
        {
          if ( *(_BYTE *)(v14 + 16) == 14 )
          {
            **(_QWORD **)(a1 + 8) = v14 + 24;
            return (unsigned int)v2;
          }
        }
      }
      goto LABEL_23;
    }
  }
  BYTE1(v9) &= ~0x80u;
  if ( (unsigned int)(v9 - 4) <= 1 )
  {
    if ( v2 != *(_BYTE **)a1 )
      goto LABEL_10;
    goto LABEL_30;
  }
LABEL_12:
  v10 = *(_BYTE **)(v4 - 48);
  v11 = *(_BYTE **)(v4 - 24);
  if ( v5 == v10 && v8 == v11 )
  {
    v12 = *(unsigned __int16 *)(v4 + 18);
  }
  else
  {
    if ( v5 != v11 || v8 != v10 )
      goto LABEL_2;
    v12 = *(unsigned __int16 *)(v4 + 18);
    if ( v5 != v10 )
    {
      v12 = sub_15FF0F0(v12 & 0xFFFF7FFF);
      goto LABEL_16;
    }
  }
  BYTE1(v12) &= ~0x80u;
LABEL_16:
  if ( (unsigned int)(v12 - 12) > 1 || v10 != *(_BYTE **)(a1 + 16) )
    goto LABEL_2;
  return sub_14B0050((_QWORD **)(a1 + 24), v11);
}
