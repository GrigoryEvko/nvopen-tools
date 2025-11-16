// Function: sub_14031A0
// Address: 0x14031a0
//
void __fastcall sub_14031A0(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // rdi
  unsigned int v6; // ecx
  _QWORD *v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // r13
  __int64 *i; // rbx
  __int64 v11; // rdi
  int v12; // r10d
  _QWORD *v13; // r9
  int v14; // eax
  int v15; // edx
  int v16; // eax
  int v17; // ecx
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // rsi
  int v21; // r10d
  _QWORD *v22; // r8
  int v23; // eax
  int v24; // eax
  __int64 v25; // rsi
  int v26; // r8d
  _QWORD *v27; // rdi
  unsigned int v28; // ebx
  __int64 v29; // rcx

  v4 = *(_DWORD *)(a2 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a2;
    goto LABEL_16;
  }
  v5 = *(_QWORD *)(a2 + 8);
  v6 = (v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v7 = (_QWORD *)(v5 + 8LL * v6);
  v8 = *v7;
  if ( a1 == *v7 )
    goto LABEL_3;
  v12 = 1;
  v13 = 0;
  while ( v8 != -8 )
  {
    if ( v13 || v8 != -16 )
      v7 = v13;
    v6 = (v4 - 1) & (v12 + v6);
    v8 = *(_QWORD *)(v5 + 8LL * v6);
    if ( a1 == v8 )
      goto LABEL_3;
    ++v12;
    v13 = v7;
    v7 = (_QWORD *)(v5 + 8LL * v6);
  }
  v14 = *(_DWORD *)(a2 + 16);
  if ( !v13 )
    v13 = v7;
  ++*(_QWORD *)a2;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v4 )
  {
LABEL_16:
    sub_1402FF0(a2, 2 * v4);
    v16 = *(_DWORD *)(a2 + 24);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a2 + 8);
      v19 = (v16 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v13 = (_QWORD *)(v18 + 8LL * v19);
      v20 = *v13;
      v15 = *(_DWORD *)(a2 + 16) + 1;
      if ( a1 != *v13 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -8 )
        {
          if ( v20 == -16 && !v22 )
            v22 = v13;
          v19 = v17 & (v21 + v19);
          v13 = (_QWORD *)(v18 + 8LL * v19);
          v20 = *v13;
          if ( a1 == *v13 )
            goto LABEL_12;
          ++v21;
        }
        if ( v22 )
          v13 = v22;
      }
      goto LABEL_12;
    }
    goto LABEL_44;
  }
  if ( v4 - *(_DWORD *)(a2 + 20) - v15 <= v4 >> 3 )
  {
    sub_1402FF0(a2, v4);
    v23 = *(_DWORD *)(a2 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a2 + 8);
      v26 = 1;
      v27 = 0;
      v28 = v24 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v13 = (_QWORD *)(v25 + 8LL * v28);
      v29 = *v13;
      v15 = *(_DWORD *)(a2 + 16) + 1;
      if ( a1 != *v13 )
      {
        while ( v29 != -8 )
        {
          if ( !v27 && v29 == -16 )
            v27 = v13;
          v28 = v24 & (v26 + v28);
          v13 = (_QWORD *)(v25 + 8LL * v28);
          v29 = *v13;
          if ( a1 == *v13 )
            goto LABEL_12;
          ++v26;
        }
        if ( v27 )
          v13 = v27;
      }
      goto LABEL_12;
    }
LABEL_44:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
LABEL_12:
  *(_DWORD *)(a2 + 16) = v15;
  if ( *v13 != -8 )
    --*(_DWORD *)(a2 + 20);
  *v13 = a1;
LABEL_3:
  nullsub_529();
  v9 = *(__int64 **)(a1 + 16);
  for ( i = *(__int64 **)(a1 + 8); v9 != i; ++i )
  {
    v11 = *i;
    sub_14031A0(v11, a2);
  }
}
