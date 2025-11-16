// Function: sub_3506D90
// Address: 0x3506d90
//
unsigned __int64 __fastcall sub_3506D90(__int64 a1, __int64 *a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  __int64 i; // rsi
  int v11; // edx
  __int64 v12; // rsi
  unsigned int v13; // r10d
  __int64 *v14; // rdx
  __int64 v15; // rbx
  int v17; // edx
  int v18; // r12d

  v8 = *(_QWORD *)(a4 + 16);
  v9 = v8;
  if ( (*(_DWORD *)(v8 + 44) & 4) != 0 )
  {
    do
      v9 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v9 + 44) & 4) != 0 );
  }
  for ( ; (*(_BYTE *)(v8 + 44) & 8) != 0; v8 = *(_QWORD *)(v8 + 8) )
    ;
  for ( i = *(_QWORD *)(v8 + 8); i != v9; v9 = *(_QWORD *)(v9 + 8) )
  {
    v11 = *(unsigned __int16 *)(v9 + 68);
    v8 = (unsigned int)(v11 - 14);
    if ( (unsigned __int16)(v11 - 14) > 4u && (_WORD)v11 != 24 )
      break;
  }
  v12 = *(_QWORD *)(a1 + 128);
  v13 = *(_DWORD *)(a1 + 144);
  if ( !v13 )
  {
LABEL_14:
    v14 = (__int64 *)(v12 + 16LL * v13);
    return sub_2E0E0B0(
             a3,
             ((*(_BYTE *)(a4 + 4) & 4) == 0 ? 4LL : 2LL) | v14[1] & 0xFFFFFFFFFFFFFFF8LL,
             a2,
             v8,
             (__int64)a2,
             a4);
  }
  v8 = (v13 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v14 = (__int64 *)(v12 + 16 * v8);
  v15 = *v14;
  if ( *v14 != v9 )
  {
    v17 = 1;
    while ( v15 != -4096 )
    {
      v18 = v17 + 1;
      v8 = (v13 - 1) & (v17 + (_DWORD)v8);
      v14 = (__int64 *)(v12 + 16LL * (unsigned int)v8);
      v15 = *v14;
      if ( *v14 == v9 )
        return sub_2E0E0B0(
                 a3,
                 ((*(_BYTE *)(a4 + 4) & 4) == 0 ? 4LL : 2LL) | v14[1] & 0xFFFFFFFFFFFFFFF8LL,
                 a2,
                 v8,
                 (__int64)a2,
                 a4);
      v17 = v18;
    }
    goto LABEL_14;
  }
  return sub_2E0E0B0(
           a3,
           ((*(_BYTE *)(a4 + 4) & 4) == 0 ? 4LL : 2LL) | v14[1] & 0xFFFFFFFFFFFFFFF8LL,
           a2,
           v8,
           (__int64)a2,
           a4);
}
