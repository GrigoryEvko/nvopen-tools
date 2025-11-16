// Function: sub_335F830
// Address: 0x335f830
//
_QWORD *__fastcall sub_335F830(__int64 a1, __int64 *a2, _QWORD *a3)
{
  unsigned int v5; // esi
  int v7; // eax
  int v8; // eax
  int v10; // eax
  __int64 v11; // rdx
  int v12; // esi
  __int64 v13; // r9
  unsigned int v14; // ecx
  __int64 v15; // rdi
  int v16; // r12d
  _QWORD *v17; // r10
  int v18; // eax
  __int64 v19; // rsi
  int v20; // ecx
  __int64 v21; // r9
  int v22; // r12d
  unsigned int v23; // edx
  __int64 v24; // rdi

  v5 = *(_DWORD *)(a1 + 24);
  v7 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v8 = v7 + 1;
  if ( 4 * v8 >= 3 * v5 )
  {
    sub_335F610(a1, 2 * v5);
    v10 = *(_DWORD *)(a1 + 24);
    if ( v10 )
    {
      v11 = *a2;
      v12 = v10 - 1;
      v13 = *(_QWORD *)(a1 + 8);
      v14 = (v10 - 1) & (37 * *a2);
      v8 = *(_DWORD *)(a1 + 16) + 1;
      a3 = (_QWORD *)(v13 + 16LL * v14);
      v15 = *a3;
      if ( *a2 == *a3 )
        goto LABEL_3;
      v16 = 1;
      v17 = 0;
      while ( v15 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( v15 == 0x8000000000000000LL && !v17 )
          v17 = a3;
        v14 = v12 & (v16 + v14);
        a3 = (_QWORD *)(v13 + 16LL * v14);
        v15 = *a3;
        if ( v11 == *a3 )
          goto LABEL_3;
        ++v16;
      }
      goto LABEL_10;
    }
LABEL_26:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v8 > v5 >> 3 )
    goto LABEL_3;
  sub_335F610(a1, v5);
  v18 = *(_DWORD *)(a1 + 24);
  if ( !v18 )
    goto LABEL_26;
  v19 = *a2;
  v20 = v18 - 1;
  v21 = *(_QWORD *)(a1 + 8);
  v17 = 0;
  v22 = 1;
  v23 = (v18 - 1) & (37 * v19);
  v8 = *(_DWORD *)(a1 + 16) + 1;
  a3 = (_QWORD *)(v21 + 16LL * v23);
  v24 = *a3;
  if ( v19 == *a3 )
    goto LABEL_3;
  while ( v24 != 0x7FFFFFFFFFFFFFFFLL )
  {
    if ( v24 == 0x8000000000000000LL && !v17 )
      v17 = a3;
    v23 = v20 & (v22 + v23);
    a3 = (_QWORD *)(v21 + 16LL * v23);
    v24 = *a3;
    if ( v19 == *a3 )
      goto LABEL_3;
    ++v22;
  }
LABEL_10:
  if ( v17 )
    a3 = v17;
LABEL_3:
  *(_DWORD *)(a1 + 16) = v8;
  if ( *a3 != 0x7FFFFFFFFFFFFFFFLL )
    --*(_DWORD *)(a1 + 20);
  return a3;
}
