// Function: sub_1603AD0
// Address: 0x1603ad0
//
_QWORD *__fastcall sub_1603AD0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned int v4; // esi
  __int64 v5; // r9
  __int64 v6; // rdi
  unsigned int v7; // ecx
  _QWORD *v8; // rax
  __int64 v9; // rdx
  _QWORD *result; // rax
  int v11; // r11d
  _QWORD *v12; // r10
  int v13; // ecx
  int v14; // ecx
  _QWORD *v15; // rdx
  int v16; // eax
  int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // edx
  __int64 v20; // rdi
  int v21; // r10d
  _QWORD *v22; // r9
  int v23; // eax
  int v24; // edx
  __int64 v25; // rdi
  _QWORD *v26; // r8
  unsigned int v27; // r13d
  int v28; // r9d
  __int64 v29; // rsi

  v3 = *a1;
  v4 = *(_DWORD *)(*a1 + 2952);
  v5 = *a1 + 2928;
  if ( !v4 )
  {
    ++*(_QWORD *)(v3 + 2928);
    goto LABEL_14;
  }
  v6 = *(_QWORD *)(v3 + 2936);
  v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (_QWORD *)(v6 + 40LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
    return v8 + 1;
  v11 = 1;
  v12 = 0;
  while ( v9 != -8 )
  {
    if ( !v12 && v9 == -16 )
      v12 = v8;
    v7 = (v4 - 1) & (v11 + v7);
    v8 = (_QWORD *)(v6 + 40LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
      return v8 + 1;
    ++v11;
  }
  v13 = *(_DWORD *)(v3 + 2944);
  if ( v12 )
    v8 = v12;
  ++*(_QWORD *)(v3 + 2928);
  v14 = v13 + 1;
  if ( 4 * v14 >= 3 * v4 )
  {
LABEL_14:
    sub_1603400(v5, 2 * v4);
    v16 = *(_DWORD *)(v3 + 2952);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(v3 + 2936);
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(v3 + 2944) + 1;
      v8 = (_QWORD *)(v18 + 40LL * v19);
      v20 = *v8;
      if ( a2 != *v8 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -8 )
        {
          if ( !v22 && v20 == -16 )
            v22 = v8;
          v19 = v17 & (v21 + v19);
          v8 = (_QWORD *)(v18 + 40LL * v19);
          v20 = *v8;
          if ( a2 == *v8 )
            goto LABEL_10;
          ++v21;
        }
        if ( v22 )
          v8 = v22;
      }
      goto LABEL_10;
    }
    goto LABEL_42;
  }
  if ( v4 - *(_DWORD *)(v3 + 2948) - v14 <= v4 >> 3 )
  {
    sub_1603400(v5, v4);
    v23 = *(_DWORD *)(v3 + 2952);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(v3 + 2936);
      v26 = 0;
      v27 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = 1;
      v14 = *(_DWORD *)(v3 + 2944) + 1;
      v8 = (_QWORD *)(v25 + 40LL * v27);
      v29 = *v8;
      if ( a2 != *v8 )
      {
        while ( v29 != -8 )
        {
          if ( !v26 && v29 == -16 )
            v26 = v8;
          v27 = v24 & (v28 + v27);
          v8 = (_QWORD *)(v25 + 40LL * v27);
          v29 = *v8;
          if ( a2 == *v8 )
            goto LABEL_10;
          ++v28;
        }
        if ( v26 )
          v8 = v26;
      }
      goto LABEL_10;
    }
LABEL_42:
    ++*(_DWORD *)(v3 + 2944);
    BUG();
  }
LABEL_10:
  *(_DWORD *)(v3 + 2944) = v14;
  if ( *v8 != -8 )
    --*(_DWORD *)(v3 + 2948);
  v15 = v8 + 3;
  *v8 = a2;
  result = v8 + 1;
  *result = v15;
  result[1] = 0;
  *((_BYTE *)result + 16) = 0;
  return result;
}
