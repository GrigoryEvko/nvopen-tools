// Function: sub_27D1DE0
// Address: 0x27d1de0
//
_QWORD *__fastcall sub_27D1DE0(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  unsigned int v5; // esi
  int v7; // eax
  int v8; // eax
  int v10; // eax
  int v11; // ecx
  __int64 v12; // r9
  unsigned int v13; // edx
  __int64 v14; // rdi
  int v15; // r11d
  _QWORD *v16; // r10
  int v17; // eax
  int v18; // ecx
  __int64 v19; // r9
  int v20; // r11d
  unsigned int v21; // edx
  __int64 v22; // rdi

  v5 = *(_DWORD *)(a1 + 24);
  v7 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v8 = v7 + 1;
  if ( 4 * v8 >= 3 * v5 )
  {
    sub_CE2A30(a1, 2 * v5);
    v10 = *(_DWORD *)(a1 + 24);
    if ( v10 )
    {
      v11 = v10 - 1;
      v12 = *(_QWORD *)(a1 + 8);
      v13 = (v10 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      a3 = (_QWORD *)(v12 + 8LL * v13);
      v14 = *a3;
      v8 = *(_DWORD *)(a1 + 16) + 1;
      if ( *a2 == *a3 )
        goto LABEL_3;
      v15 = 1;
      v16 = 0;
      while ( v14 != -4096 )
      {
        if ( v14 == -8192 && !v16 )
          v16 = a3;
        v13 = v11 & (v15 + v13);
        a3 = (_QWORD *)(v12 + 8LL * v13);
        v14 = *a3;
        if ( *a2 == *a3 )
          goto LABEL_3;
        ++v15;
      }
      goto LABEL_10;
    }
LABEL_26:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v8 > v5 >> 3 )
    goto LABEL_3;
  sub_CE2A30(a1, v5);
  v17 = *(_DWORD *)(a1 + 24);
  if ( !v17 )
    goto LABEL_26;
  v18 = v17 - 1;
  v19 = *(_QWORD *)(a1 + 8);
  v16 = 0;
  v20 = 1;
  v21 = (v17 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  a3 = (_QWORD *)(v19 + 8LL * v21);
  v22 = *a3;
  v8 = *(_DWORD *)(a1 + 16) + 1;
  if ( *a2 == *a3 )
    goto LABEL_3;
  while ( v22 != -4096 )
  {
    if ( v22 == -8192 && !v16 )
      v16 = a3;
    v21 = v18 & (v20 + v21);
    a3 = (_QWORD *)(v19 + 8LL * v21);
    v22 = *a3;
    if ( *a2 == *a3 )
      goto LABEL_3;
    ++v20;
  }
LABEL_10:
  if ( v16 )
    a3 = v16;
LABEL_3:
  *(_DWORD *)(a1 + 16) = v8;
  if ( *a3 != -4096 )
    --*(_DWORD *)(a1 + 20);
  return a3;
}
