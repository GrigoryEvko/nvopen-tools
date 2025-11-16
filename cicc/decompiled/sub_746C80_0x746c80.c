// Function: sub_746C80
// Address: 0x746c80
//
__int64 __fastcall sub_746C80(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 *v8; // rax
  char v9; // si
  __int64 *v10; // rdi
  unsigned __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rdx
  unsigned __int64 v14; // rcx
  bool v15; // al
  __int64 v16; // rdx
  __int64 v17; // rax
  char i; // dl
  unsigned int (*v19)(void); // rax

  v4 = *(unsigned __int8 *)(a1 + 184);
  if ( !*(_BYTE *)(a2 + 136) )
  {
    if ( (unsigned __int8)v4 > 0xAu )
      goto LABEL_11;
    v7 = 1821;
    if ( !_bittest64(&v7, v4) )
      goto LABEL_11;
LABEL_8:
    if ( !*(_BYTE *)(a2 + 149) )
      return 0;
    return (unsigned __int8)(v4 - 2) <= 2u;
  }
  if ( !*(_BYTE *)(a2 + 141) )
  {
    if ( (unsigned __int8)v4 > 0xAu )
      return 1;
    v13 = 1821;
    if ( !_bittest64(&v13, v4) )
      return 1;
    goto LABEL_8;
  }
  if ( (unsigned __int8)v4 <= 0xAu )
  {
    v5 = 1821;
    if ( _bittest64(&v5, v4) )
      return 0;
  }
LABEL_11:
  v8 = sub_746BE0(a1);
  v9 = *(_BYTE *)(a2 + 136);
  v10 = v8;
  if ( !v9 )
  {
    v14 = *(unsigned __int8 *)(a1 + 184);
    if ( (unsigned __int8)v14 <= 0xAu )
    {
      v15 = ((0x71DuLL >> v14) & 1) == 0;
    }
    else
    {
      v15 = 1;
      if ( (unsigned __int8)v14 > 0xCu )
        return (*(_BYTE *)(a1 + 186) & 8) != 0;
    }
    v16 = 6338;
    if ( (_bittest64(&v16, v14) || !v15) && (unsigned __int8)(v14 - 11) > 1u && (v10 || (_BYTE)v14 == 1) )
      goto LABEL_29;
    return (*(_BYTE *)(a1 + 186) & 8) != 0;
  }
  if ( *(_BYTE *)(a2 + 141) )
    return 0;
  v11 = *(unsigned __int8 *)(a1 + 184);
  if ( (unsigned __int8)v11 <= 0xAu )
  {
    LOBYTE(result) = ((0x71DuLL >> v11) & 1) == 0;
LABEL_15:
    v12 = 6338;
    if ( !_bittest64(&v12, v11) && (_BYTE)result || (unsigned __int8)(v11 - 11) <= 1u || !v10 && (_BYTE)v11 != 1 )
      return 1;
LABEL_29:
    v17 = *(_QWORD *)(a1 + 160);
    for ( i = *(_BYTE *)(v17 + 140); i == 12; i = *(_BYTE *)(v17 + 140) )
      v17 = *(_QWORD *)(v17 + 160);
    if ( i == 14 && *(_BYTE *)(v17 + 160) == 2 )
      return 1;
    if ( v9 && v10 )
    {
      v19 = *(unsigned int (**)(void))(a2 + 112);
      if ( v19 )
        return v19() == 0;
      return 1;
    }
    return 0;
  }
  result = 1;
  if ( (unsigned __int8)v11 <= 0xCu )
    goto LABEL_15;
  return result;
}
