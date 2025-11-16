// Function: sub_1A108C0
// Address: 0x1a108c0
//
__int64 __fastcall sub_1A108C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  _QWORD *v8; // r9
  _QWORD *v9; // r8
  unsigned int v10; // ecx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 result; // rax
  int v16; // r15d
  _QWORD *v17; // r10
  int v18; // ecx
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  unsigned int v22; // edx
  __int64 v23; // rdi
  int v24; // r10d
  int v25; // eax
  int v26; // edx
  __int64 v27; // rdi
  unsigned int v28; // r14d
  __int64 v29; // rsi
  __int64 *v30; // r11

  v6 = a1 + 120;
  v7 = *(_DWORD *)(a1 + 144);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_21;
  }
  LODWORD(v8) = v7 - 1;
  v9 = *(_QWORD **)(a1 + 128);
  v10 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = &v9[2 * v10];
  v12 = *v11;
  if ( *v11 == a2 )
  {
    v13 = v11[1] & 1LL;
    goto LABEL_4;
  }
  v16 = 1;
  v17 = 0;
  while ( v12 != -8 )
  {
    if ( v12 != -16 || v17 )
      v11 = v17;
    v10 = (unsigned int)v8 & (v16 + v10);
    v30 = &v9[2 * v10];
    v12 = *v30;
    if ( *v30 == a2 )
    {
      v11 = &v9[2 * v10];
      v13 = v30[1] & 1;
      goto LABEL_4;
    }
    ++v16;
    v17 = v11;
    v11 = &v9[2 * v10];
  }
  v18 = *(_DWORD *)(a1 + 136);
  if ( v17 )
    v11 = v17;
  ++*(_QWORD *)(a1 + 120);
  v19 = v18 + 1;
  if ( 4 * v19 >= 3 * v7 )
  {
LABEL_21:
    sub_1A0FE70(v6, 2 * v7);
    v20 = *(_DWORD *)(a1 + 144);
    if ( v20 )
    {
      v21 = v20 - 1;
      v9 = *(_QWORD **)(a1 + 128);
      v22 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = *(_DWORD *)(a1 + 136) + 1;
      v11 = &v9[2 * v22];
      v23 = *v11;
      if ( *v11 != a2 )
      {
        v24 = 1;
        v8 = 0;
        while ( v23 != -8 )
        {
          if ( !v8 && v23 == -16 )
            v8 = v11;
          v22 = v21 & (v24 + v22);
          v11 = &v9[2 * v22];
          v23 = *v11;
          if ( *v11 == a2 )
            goto LABEL_17;
          ++v24;
        }
        if ( v8 )
          v11 = v8;
      }
      goto LABEL_17;
    }
    goto LABEL_50;
  }
  LODWORD(v9) = v7 >> 3;
  if ( v7 - *(_DWORD *)(a1 + 140) - v19 <= v7 >> 3 )
  {
    sub_1A0FE70(v6, v7);
    v25 = *(_DWORD *)(a1 + 144);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 128);
      v9 = 0;
      v28 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      LODWORD(v8) = 1;
      v19 = *(_DWORD *)(a1 + 136) + 1;
      v11 = (_QWORD *)(v27 + 16LL * v28);
      v29 = *v11;
      if ( *v11 != a2 )
      {
        while ( v29 != -8 )
        {
          if ( v29 == -16 && !v9 )
            v9 = v11;
          v28 = v26 & ((_DWORD)v8 + v28);
          v11 = (_QWORD *)(v27 + 16LL * v28);
          v29 = *v11;
          if ( *v11 == a2 )
            goto LABEL_17;
          LODWORD(v8) = (_DWORD)v8 + 1;
        }
        if ( v9 )
          v11 = v9;
      }
      goto LABEL_17;
    }
LABEL_50:
    ++*(_DWORD *)(a1 + 136);
    BUG();
  }
LABEL_17:
  *(_DWORD *)(a1 + 136) = v19;
  if ( *v11 != -8 )
    --*(_DWORD *)(a1 + 140);
  *v11 = a2;
  v13 = 0;
  v11[1] = 0;
LABEL_4:
  v14 = a3 | v13 | 4;
  v11[1] = v14;
  if ( (((unsigned __int8)v14 ^ 6) & 6) != 0 )
  {
    result = *(unsigned int *)(a1 + 1352);
    if ( (unsigned int)result >= *(_DWORD *)(a1 + 1356) )
    {
      sub_16CD150(a1 + 1344, (const void *)(a1 + 1360), 0, 8, (int)v9, (int)v8);
      result = *(unsigned int *)(a1 + 1352);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 1344) + 8 * result) = a2;
    ++*(_DWORD *)(a1 + 1352);
  }
  else
  {
    result = *(unsigned int *)(a1 + 824);
    if ( (unsigned int)result >= *(_DWORD *)(a1 + 828) )
    {
      sub_16CD150(a1 + 816, (const void *)(a1 + 832), 0, 8, (int)v9, (int)v8);
      result = *(unsigned int *)(a1 + 824);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 816) + 8 * result) = a2;
    ++*(_DWORD *)(a1 + 824);
  }
  return result;
}
