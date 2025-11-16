// Function: sub_3734CF0
// Address: 0x3734cf0
//
unsigned __int64 __fastcall sub_3734CF0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  int v6; // r10d
  __int64 v7; // r8
  __int64 *v8; // rax
  unsigned int v9; // ecx
  __int64 v10; // rbx
  __int64 v11; // rdx
  _DWORD *v12; // rax
  int v14; // ecx
  int v15; // ecx
  int v16; // eax
  int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // edx
  __int64 v20; // rdi
  int v21; // r10d
  __int64 *v22; // r9
  int v23; // eax
  int v24; // edx
  int v25; // r9d
  __int64 *v26; // r8
  __int64 v27; // rdi
  unsigned int v28; // r14d
  __int64 v29; // rsi

  v4 = a1 + 168;
  v5 = *(_DWORD *)(a1 + 192);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 168);
    goto LABEL_20;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 176);
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = v7 + 16LL * v9;
  v11 = *(_QWORD *)v10;
  if ( a2 != *(_QWORD *)v10 )
  {
    while ( v11 != -4096 )
    {
      if ( !v8 && v11 == -8192 )
        v8 = (__int64 *)v10;
      v9 = (v5 - 1) & (v6 + v9);
      v10 = v7 + 16LL * v9;
      v11 = *(_QWORD *)v10;
      if ( a2 == *(_QWORD *)v10 )
        goto LABEL_3;
      ++v6;
    }
    v14 = *(_DWORD *)(a1 + 184);
    if ( !v8 )
      v8 = (__int64 *)v10;
    ++*(_QWORD *)(a1 + 168);
    v15 = v14 + 1;
    if ( 4 * v15 < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 188) - v15 > v5 >> 3 )
      {
LABEL_15:
        *(_DWORD *)(a1 + 184) = v15;
        if ( *v8 != -4096 )
          --*(_DWORD *)(a1 + 188);
        *v8 = a2;
        v12 = v8 + 1;
        *v12 = 0;
        goto LABEL_18;
      }
      sub_3733670(v4, v5);
      v23 = *(_DWORD *)(a1 + 192);
      if ( v23 )
      {
        v24 = v23 - 1;
        v25 = 1;
        v26 = 0;
        v27 = *(_QWORD *)(a1 + 176);
        v28 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v15 = *(_DWORD *)(a1 + 184) + 1;
        v8 = (__int64 *)(v27 + 16LL * v28);
        v29 = *v8;
        if ( a2 != *v8 )
        {
          while ( v29 != -4096 )
          {
            if ( !v26 && v29 == -8192 )
              v26 = v8;
            v28 = v24 & (v25 + v28);
            v8 = (__int64 *)(v27 + 16LL * v28);
            v29 = *v8;
            if ( a2 == *v8 )
              goto LABEL_15;
            ++v25;
          }
          if ( v26 )
            v8 = v26;
        }
        goto LABEL_15;
      }
LABEL_43:
      ++*(_DWORD *)(a1 + 184);
      BUG();
    }
LABEL_20:
    sub_3733670(v4, 2 * v5);
    v16 = *(_DWORD *)(a1 + 192);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 176);
      v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 184) + 1;
      v8 = (__int64 *)(v18 + 16LL * v19);
      v20 = *v8;
      if ( a2 != *v8 )
      {
        v21 = 1;
        v22 = 0;
        while ( v20 != -4096 )
        {
          if ( !v22 && v20 == -8192 )
            v22 = v8;
          v19 = v17 & (v21 + v19);
          v8 = (__int64 *)(v18 + 16LL * v19);
          v20 = *v8;
          if ( a2 == *v8 )
            goto LABEL_15;
          ++v21;
        }
        if ( v22 )
          v8 = v22;
      }
      goto LABEL_15;
    }
    goto LABEL_43;
  }
LABEL_3:
  v12 = (_DWORD *)(v10 + 8);
  if ( *(_DWORD *)(v10 + 8) )
  {
    sub_372FCB0((int *)a1, 0x52u);
    return sub_372FCB0((int *)a1, *(unsigned int *)(v10 + 8));
  }
LABEL_18:
  *v12 = *(_DWORD *)(a1 + 184);
  sub_372FCB0((int *)a1, 0x54u);
  return sub_3734910((int *)a1, a2);
}
