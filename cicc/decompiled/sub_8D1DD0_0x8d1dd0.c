// Function: sub_8D1DD0
// Address: 0x8d1dd0
//
_BOOL8 __fastcall sub_8D1DD0(__int64 a1, __int64 a2, char a3)
{
  _BOOL4 v3; // r8d
  unsigned int v5; // r13d
  char v7; // al
  unsigned __int8 v8; // cl
  unsigned __int64 v9; // rdx
  unsigned __int8 v10; // al
  unsigned __int8 v11; // cl
  unsigned __int64 v13; // rsi
  char v14; // dl
  __int64 *v15; // r14
  __int64 *v16; // rax
  __int128 v17; // rdi
  int v18; // eax

  v3 = 0;
  v5 = (a3 & 0x40) == 0 ? 4 : 6;
  while ( 1 )
  {
    v7 = *(_BYTE *)(a2 + 140);
    if ( *(_BYTE *)(a1 + 140) != 12 )
    {
      if ( v7 == 12 )
      {
        LODWORD(v9) = 0;
        goto LABEL_8;
      }
      return v3;
    }
    while ( 1 )
    {
      if ( (*(_BYTE *)(a1 + 186) & 8) != 0 )
      {
        v8 = *(_BYTE *)(a1 + 184);
        if ( v8 != 7 )
          break;
      }
      a1 = *(_QWORD *)(a1 + 160);
      if ( *(_BYTE *)(a1 + 140) != 12 )
      {
        LODWORD(v9) = 0;
        goto LABEL_7;
      }
    }
    LODWORD(v9) = 0;
    if ( v8 <= 0xCu )
      v9 = (0x1842uLL >> v8) & 1;
LABEL_7:
    if ( v7 != 12 )
    {
LABEL_11:
      if ( (_DWORD)v9 )
        return 1;
      return v3;
    }
LABEL_8:
    while ( 1 )
    {
      v10 = *(_BYTE *)(a2 + 186);
      if ( (v10 & 8) != 0 )
      {
        v11 = *(_BYTE *)(a2 + 184);
        if ( v11 != 7 )
          break;
      }
      a2 = *(_QWORD *)(a2 + 160);
      if ( *(_BYTE *)(a2 + 140) != 12 )
        goto LABEL_11;
    }
    if ( v11 > 0xCu )
      break;
    v13 = (0x1842uLL >> v11) & 1;
    if ( !(_DWORD)v9 )
    {
      if ( !(_BYTE)v13 )
        return v3;
      return 1;
    }
    if ( !(_BYTE)v13 )
      return 1;
    if ( a1 == a2 )
    {
      v3 = 0;
    }
    else
    {
      v14 = *(_BYTE *)(a1 + 184);
      if ( (v14 == 1) != (v11 == 1) )
        return 1;
      if ( (v14 == 6) != (v11 == 6) )
        return 1;
      if ( v14 == 7 )
        return 1;
      if ( ((*(_BYTE *)(a1 + 186) ^ v10) & 2) != 0 )
        return 1;
      v15 = sub_746BE0(a1);
      v16 = sub_746BE0(a2);
      *((_QWORD *)&v17 + 1) = v16;
      if ( !v15 || !v16 )
        return 1;
      *(_QWORD *)&v17 = v15;
      v18 = sub_7386E0(v17, v5);
      v3 = v18 == 0;
      if ( !v18 )
        return v3;
    }
    a1 = *(_QWORD *)(a1 + 160);
    a2 = *(_QWORD *)(a2 + 160);
  }
  if ( (_DWORD)v9 )
    return 1;
  return v3;
}
