// Function: sub_5E5070
// Address: 0x5e5070
//
__int64 __fastcall sub_5E5070(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, _BOOL4 a5, __int64 a6)
{
  __int64 j; // r12
  char i; // al
  __int64 v12; // rcx
  __int64 v14; // rbx
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rbx
  char v20; // r13
  __int64 v21; // rax
  unsigned int v22; // [rsp+Ch] [rbp-34h]

  j = a1;
  for ( i = *(_BYTE *)(a1 + 140); i == 12; i = *(_BYTE *)(j + 140) )
    j = *(_QWORD *)(j + 160);
  if ( i == 8 )
  {
    v22 = a3;
    v16 = sub_8D40F0(j);
    a3 = v22;
    for ( j = v16; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
  }
  while ( *(_BYTE *)(a2 + 140) == 12 )
    a2 = *(_QWORD *)(a2 + 160);
  v12 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  if ( dword_4F077BC || dword_4F077B4 )
  {
    if ( a5 )
      *(_BYTE *)(v12 + 182) |= 4u;
  }
  else if ( a5 )
  {
    *(_BYTE *)(v12 + 182) = *(_BYTE *)(v12 + 182) & 0xF3 | 4;
  }
  else
  {
    a5 = (*(_BYTE *)(v12 + 182) & 4) != 0;
  }
  if ( (unsigned __int8)(*(_BYTE *)(j + 140) - 9) > 2u || (*(_BYTE *)(j + 177) & 0x20) != 0 )
    return 1;
  v14 = *(_QWORD *)(*(_QWORD *)j + 96LL);
  if ( dword_4D04434 )
  {
    if ( (_DWORD)a3 )
    {
      if ( (*(_BYTE *)(v14 + 182) & 8) == 0 )
      {
LABEL_19:
        if ( *(char *)(v14 + 177) < 0 )
          *(_BYTE *)(v12 + 182) |= 0x10u;
        if ( (*(_BYTE *)(v14 + 178) & 1) != 0 )
          *(_BYTE *)(v12 + 182) |= 0x20u;
        if ( *(_QWORD *)(v14 + 24) && (*(_BYTE *)(v14 + 177) & 2) == 0 )
          *(_BYTE *)(v12 + 182) |= 0x40u;
        v15 = *(_BYTE *)(v14 + 178);
        if ( (v15 & 2) != 0 )
        {
          *(_BYTE *)(v12 + 182) |= 0x80u;
          v15 = *(_BYTE *)(v14 + 178);
        }
        if ( (v15 & 4) != 0 )
          *(_BYTE *)(v12 + 183) |= 1u;
        return 1;
      }
    }
    else if ( (*(_BYTE *)(v14 + 176) & 1) == 0 || a5 )
    {
      goto LABEL_19;
    }
    *(_BYTE *)(v12 + 182) |= 8u;
    goto LABEL_19;
  }
  if ( *(_QWORD *)(v14 + 8) && (unsigned int)sub_879360(*(_QWORD *)(*(_QWORD *)j + 96LL), dword_4D04434, a3, v12)
    || *(_QWORD *)(v14 + 24) && (*(_BYTE *)(v14 + 177) & 2) == 0 )
  {
    goto LABEL_35;
  }
  if ( (*(_BYTE *)(v14 + 178) & 6) == 0 )
    return 1;
  v19 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)j + 96LL) + 32LL);
  if ( !v19 )
    return 1;
  v20 = *(_BYTE *)(v19 + 80);
  if ( v20 == 17 )
  {
    v19 = *(_QWORD *)(v19 + 88);
    if ( !v19 )
      return 1;
  }
  while ( 1 )
  {
    v21 = *(_QWORD *)(v19 + 88);
    if ( (*(_BYTE *)(v21 + 194) & 4) == 0 )
    {
      if ( (unsigned int)sub_72F5E0(*(_QWORD *)(v21 + 152), j, 1, 0, 0, 0) )
        break;
    }
    if ( v20 == 17 )
    {
      v19 = *(_QWORD *)(v19 + 8);
      if ( v19 )
        continue;
    }
    return 1;
  }
  if ( !unk_4D04960 )
  {
LABEL_35:
    v17 = (unsigned int)(a4 == 0) + 7;
    v18 = a4 == 0 ? 294 : 1398;
  }
  else if ( a4 )
  {
    v17 = 5;
    v18 = 1398;
  }
  else
  {
    a4 = 1;
    v17 = 5;
    v18 = 294;
  }
  sub_685260(v17, v18, a6, j);
  return a4;
}
