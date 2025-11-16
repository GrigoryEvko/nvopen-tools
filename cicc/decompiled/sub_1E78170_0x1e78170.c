// Function: sub_1E78170
// Address: 0x1e78170
//
__int64 __fastcall sub_1E78170(__int64 a1, int a2, __int64 a3, __int64 a4, _BYTE *a5, _BYTE *a6)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // r14
  __int64 v16; // r13

  *a5 = 1;
  v7 = *(_QWORD *)(a1 + 248);
  if ( a2 < 0 )
    v8 = *(_QWORD *)(*(_QWORD *)(v7 + 24) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v8 = *(_QWORD *)(*(_QWORD *)(v7 + 272) + 8LL * (unsigned int)a2);
  while ( 1 )
  {
    if ( !v8 )
      return 1;
    if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 && (*(_BYTE *)(v8 + 4) & 8) == 0 )
      break;
    v8 = *(_QWORD *)(v8 + 32);
  }
LABEL_8:
  v10 = *(_QWORD *)(v8 + 16);
  if ( *(_QWORD *)(v10 + 24) == a3
    && (!**(_WORD **)(v10 + 16) || **(_WORD **)(v10 + 16) == 45)
    && *(_QWORD *)(*(_QWORD *)(v10 + 32)
                 + 40LL * (-858993459 * (unsigned int)((v8 - *(_QWORD *)(v10 + 32)) >> 3) + 1)
                 + 24) == a4 )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v8 + 32);
      if ( !v8 )
        return 1;
      if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 && (*(_BYTE *)(v8 + 4) & 8) == 0 )
        goto LABEL_8;
    }
  }
  *a5 = 0;
  v11 = *(_QWORD *)(a1 + 248);
  if ( a2 < 0 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(v11 + 24) + 16LL * (a2 & 0x7FFFFFFF) + 8);
    goto LABEL_16;
  }
  v12 = *(_QWORD *)(*(_QWORD *)(v11 + 272) + 8LL * (unsigned int)a2);
  if ( !v12 )
    return 1;
  while ( (*(_BYTE *)(v12 + 3) & 0x10) != 0 || (*(_BYTE *)(v12 + 4) & 8) != 0 )
  {
    v12 = *(_QWORD *)(v12 + 32);
LABEL_16:
    if ( !v12 )
      return 1;
  }
  v13 = *(_QWORD *)(v12 + 16);
  v14 = **(unsigned __int16 **)(v13 + 16);
  if ( !**(_WORD **)(v13 + 16) )
    goto LABEL_26;
LABEL_19:
  if ( v14 == 45 )
  {
LABEL_26:
    v15 = *(_QWORD *)(*(_QWORD *)(v13 + 32)
                    + 40LL * (-858993459 * (unsigned int)((v12 - *(_QWORD *)(v13 + 32)) >> 3) + 1)
                    + 24);
    goto LABEL_21;
  }
  v15 = *(_QWORD *)(v13 + 24);
  if ( a4 == v15 )
  {
    *a6 = 1;
    return 0;
  }
LABEL_21:
  v16 = *(_QWORD *)(a1 + 256);
  sub_1E06620(v16);
  if ( sub_1E05550(*(_QWORD *)(v16 + 1312), a3, v15) )
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(v12 + 32);
      if ( !v12 )
        return 1;
      if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 && (*(_BYTE *)(v12 + 4) & 8) == 0 )
      {
        v13 = *(_QWORD *)(v12 + 16);
        v14 = **(unsigned __int16 **)(v13 + 16);
        if ( **(_WORD **)(v13 + 16) )
          goto LABEL_19;
        goto LABEL_26;
      }
    }
  }
  return 0;
}
