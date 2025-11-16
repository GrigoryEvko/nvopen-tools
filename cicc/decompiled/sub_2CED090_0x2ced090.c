// Function: sub_2CED090
// Address: 0x2ced090
//
__int64 __fastcall sub_2CED090(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r10
  __int64 result; // rax
  int v13; // edx
  unsigned int v14; // esi
  __int64 v15; // r9
  int v16; // r14d
  unsigned __int64 *v17; // rdi
  unsigned int v18; // r8d
  __int64 *v19; // rdx
  __int64 v20; // rcx
  int v21; // r11d
  int v22; // ecx
  int v23; // ecx
  int v24; // edx
  int v25; // edx
  __int64 v26; // r9
  unsigned int v27; // esi
  unsigned __int64 v28; // r8
  int v29; // r11d
  unsigned __int64 *v30; // r10
  int v31; // edx
  int v32; // esi
  __int64 v33; // r8
  unsigned __int64 *v34; // r9
  unsigned int v35; // r13d
  int v36; // r10d
  unsigned __int64 v37; // rdx
  unsigned int v38; // [rsp+Ch] [rbp-24h]
  unsigned int v39; // [rsp+Ch] [rbp-24h]

  v7 = *(unsigned int *)(a3 + 24);
  v8 = *(_QWORD *)(a3 + 8);
  if ( (_DWORD)v7 )
  {
    v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
    {
LABEL_3:
      if ( v10 != (__int64 *)(v8 + 16 * v7) )
        return *((unsigned int *)v10 + 2);
    }
    else
    {
      v13 = 1;
      while ( v11 != -4096 )
      {
        v21 = v13 + 1;
        v9 = (v7 - 1) & (v13 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( *v10 == a2 )
          goto LABEL_3;
        v13 = v21;
      }
    }
  }
  result = sub_2CECAD0(a1, a2, a3, a4);
  v14 = *(_DWORD *)(a3 + 24);
  if ( !v14 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_26;
  }
  v15 = *(_QWORD *)(a3 + 8);
  v16 = 1;
  v17 = 0;
  v18 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v19 = (__int64 *)(v15 + 16LL * v18);
  v20 = *v19;
  if ( *v19 == a2 )
  {
LABEL_9:
    *((_DWORD *)v19 + 2) = result;
    return result;
  }
  while ( v20 != -4096 )
  {
    if ( v20 == -8192 && !v17 )
      v17 = (unsigned __int64 *)v19;
    v18 = (v14 - 1) & (v16 + v18);
    v19 = (__int64 *)(v15 + 16LL * v18);
    v20 = *v19;
    if ( *v19 == a2 )
      goto LABEL_9;
    ++v16;
  }
  v22 = *(_DWORD *)(a3 + 16);
  if ( !v17 )
    v17 = (unsigned __int64 *)v19;
  ++*(_QWORD *)a3;
  v23 = v22 + 1;
  if ( 4 * v23 >= 3 * v14 )
  {
LABEL_26:
    v38 = result;
    sub_D39D40(a3, 2 * v14);
    v24 = *(_DWORD *)(a3 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a3 + 8);
      v27 = v25 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = *(_DWORD *)(a3 + 16) + 1;
      result = v38;
      v17 = (unsigned __int64 *)(v26 + 16LL * v27);
      v28 = *v17;
      if ( *v17 != a2 )
      {
        v29 = 1;
        v30 = 0;
        while ( v28 != -4096 )
        {
          if ( v28 == -8192 && !v30 )
            v30 = v17;
          v27 = v25 & (v29 + v27);
          v17 = (unsigned __int64 *)(v26 + 16LL * v27);
          v28 = *v17;
          if ( *v17 == a2 )
            goto LABEL_22;
          ++v29;
        }
        if ( v30 )
          v17 = v30;
      }
      goto LABEL_22;
    }
    goto LABEL_49;
  }
  if ( v14 - *(_DWORD *)(a3 + 20) - v23 <= v14 >> 3 )
  {
    v39 = result;
    sub_D39D40(a3, v14);
    v31 = *(_DWORD *)(a3 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a3 + 8);
      v34 = 0;
      v35 = (v31 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v36 = 1;
      v23 = *(_DWORD *)(a3 + 16) + 1;
      result = v39;
      v17 = (unsigned __int64 *)(v33 + 16LL * v35);
      v37 = *v17;
      if ( *v17 != a2 )
      {
        while ( v37 != -4096 )
        {
          if ( !v34 && v37 == -8192 )
            v34 = v17;
          v35 = v32 & (v36 + v35);
          v17 = (unsigned __int64 *)(v33 + 16LL * v35);
          v37 = *v17;
          if ( *v17 == a2 )
            goto LABEL_22;
          ++v36;
        }
        if ( v34 )
          v17 = v34;
      }
      goto LABEL_22;
    }
LABEL_49:
    ++*(_DWORD *)(a3 + 16);
    BUG();
  }
LABEL_22:
  *(_DWORD *)(a3 + 16) = v23;
  if ( *v17 != -4096 )
    --*(_DWORD *)(a3 + 20);
  *v17 = a2;
  *((_DWORD *)v17 + 2) = 0;
  *((_DWORD *)v17 + 2) = result;
  return result;
}
