// Function: sub_326B540
// Address: 0x326b540
//
__int64 __fastcall sub_326B540(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int *v4; // rdx
  __int64 v5; // rcx
  __int64 v7; // rax
  __int64 v9; // rdx
  int v10; // esi
  __int64 v11; // rax
  unsigned __int16 *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int128 v19; // rax
  int v20; // r9d

  v4 = *(unsigned int **)(a1 + 40);
  v5 = *(_QWORD *)v4;
  if ( *(_DWORD *)(*(_QWORD *)v4 + 24LL) != 200 )
    return 0;
  v7 = *(_QWORD *)(v5 + 56);
  if ( !v7 )
    return 0;
  v9 = v4[2];
  v10 = 1;
  do
  {
    while ( *(_DWORD *)(v7 + 8) != (_DWORD)v9 )
    {
      v7 = *(_QWORD *)(v7 + 32);
      if ( !v7 )
        goto LABEL_11;
    }
    if ( !v10 )
      return 0;
    v11 = *(_QWORD *)(v7 + 32);
    if ( !v11 )
      goto LABEL_12;
    if ( (_DWORD)v9 == *(_DWORD *)(v11 + 8) )
      return 0;
    v7 = *(_QWORD *)(v11 + 32);
    v10 = 0;
  }
  while ( v7 );
LABEL_11:
  if ( v10 == 1 )
    return 0;
LABEL_12:
  v12 = *(unsigned __int16 **)(a1 + 48);
  v13 = *(_QWORD *)(a2 + 16);
  v14 = *v12;
  v15 = *((_QWORD *)v12 + 1);
  v16 = *(unsigned __int16 *)(*(_QWORD *)(v5 + 48) + 16 * v9);
  v17 = 1;
  if ( (_WORD)v16 == 1 || (_WORD)v16 && (v17 = (unsigned __int16)v16, *(_QWORD *)(v13 + 8 * v16 + 112)) )
  {
    if ( (*(_BYTE *)(v13 + 500 * v17 + 6614) & 0xFB) == 0 )
      return 0;
  }
  v18 = 1;
  if ( (_WORD)v14 != 1 )
  {
    if ( !(_WORD)v14 )
      return 0;
    v18 = (unsigned __int16)v14;
    if ( !*(_QWORD *)(v13 + 8 * v14 + 112) )
      return 0;
  }
  if ( (*(_BYTE *)(v13 + 500 * v18 + 6614) & 0xFB) != 0 )
    return 0;
  *(_QWORD *)&v19 = sub_33FB310(
                      a2,
                      **(_QWORD **)(v5 + 40),
                      *(_QWORD *)(*(_QWORD *)(v5 + 40) + 8LL),
                      a3,
                      (unsigned __int16)v14,
                      v15);
  return sub_33FAF80(a2, 200, a3, (unsigned __int16)v14, v15, v20, v19);
}
