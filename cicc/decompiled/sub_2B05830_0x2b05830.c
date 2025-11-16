// Function: sub_2B05830
// Address: 0x2b05830
//
__int64 __fastcall sub_2B05830(__int64 **a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned int v4; // esi
  __int64 v5; // rcx
  unsigned int v6; // edi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 result; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  unsigned int v13; // edx
  int v14; // r14d
  __int64 *v15; // r8
  unsigned int v16; // r9d
  __int64 *v17; // rdx
  __int64 v18; // r10
  int v19; // eax
  int v20; // edx
  int v21; // edx
  __int64 v22; // r9
  unsigned int v23; // esi
  int v24; // ecx
  __int64 v25; // rdi
  int v26; // ecx
  int v27; // edx
  int v28; // esi
  __int64 v29; // rdi
  __int64 *v30; // r9
  unsigned int v31; // r13d
  int v32; // r10d
  __int64 v33; // rdx
  int v34; // r9d
  int v35; // r11d
  __int64 *v36; // r10
  unsigned int v37; // [rsp+Ch] [rbp-24h]
  unsigned int v38; // [rsp+Ch] [rbp-24h]

  v3 = **a1;
  v4 = *(_DWORD *)(v3 + 24);
  v5 = *(_QWORD *)(v3 + 8);
  if ( !v4 )
  {
    v11 = a2 + 24;
    v12 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL);
    if ( v12 == a2 + 24 )
    {
      LODWORD(result) = 0;
      goto LABEL_17;
    }
    goto LABEL_6;
  }
  v6 = v4 - 1;
  v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( *v8 != a2 )
  {
    v19 = 1;
    while ( v9 != -4096 )
    {
      v34 = v19 + 1;
      v7 = v6 & (v19 + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == a2 )
        goto LABEL_3;
      v19 = v34;
    }
    v11 = a2 + 24;
    v12 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL);
    if ( v12 == a2 + 24 )
    {
      result = 0;
      goto LABEL_10;
    }
    goto LABEL_6;
  }
LABEL_3:
  if ( v8 != (__int64 *)(v5 + 16LL * v4) )
    return *((unsigned int *)v8 + 2);
  v11 = a2 + 24;
  v12 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL);
  if ( a2 + 24 != v12 )
  {
LABEL_6:
    v13 = 0;
    do
    {
      v12 = *(_QWORD *)(v12 + 8);
      ++v13;
    }
    while ( v12 != v11 );
    result = v13;
    if ( v4 )
    {
      v6 = v4 - 1;
      goto LABEL_10;
    }
LABEL_17:
    ++*(_QWORD *)v3;
    goto LABEL_18;
  }
  result = 0;
LABEL_10:
  v14 = 1;
  v15 = 0;
  v16 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v17 = (__int64 *)(v5 + 16LL * v16);
  v18 = *v17;
  if ( *v17 == a2 )
  {
LABEL_11:
    *((_DWORD *)v17 + 2) = result;
    return result;
  }
  while ( v18 != -4096 )
  {
    if ( v18 == -8192 && !v15 )
      v15 = v17;
    v16 = v6 & (v14 + v16);
    v17 = (__int64 *)(v5 + 16LL * v16);
    v18 = *v17;
    if ( *v17 == a2 )
      goto LABEL_11;
    ++v14;
  }
  v26 = *(_DWORD *)(v3 + 16);
  if ( !v15 )
    v15 = v17;
  ++*(_QWORD *)v3;
  v24 = v26 + 1;
  if ( 4 * v24 >= 3 * v4 )
  {
LABEL_18:
    v37 = result;
    sub_2809BA0(v3, 2 * v4);
    v20 = *(_DWORD *)(v3 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(v3 + 8);
      v23 = v21 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v24 = *(_DWORD *)(v3 + 16) + 1;
      result = v37;
      v15 = (__int64 *)(v22 + 16LL * v23);
      v25 = *v15;
      if ( *v15 != a2 )
      {
        v35 = 1;
        v36 = 0;
        while ( v25 != -4096 )
        {
          if ( !v36 && v25 == -8192 )
            v36 = v15;
          v23 = v21 & (v35 + v23);
          v15 = (__int64 *)(v22 + 16LL * v23);
          v25 = *v15;
          if ( *v15 == a2 )
            goto LABEL_20;
          ++v35;
        }
        if ( v36 )
          v15 = v36;
      }
      goto LABEL_20;
    }
    goto LABEL_58;
  }
  if ( v4 - *(_DWORD *)(v3 + 20) - v24 <= v4 >> 3 )
  {
    v38 = result;
    sub_2809BA0(v3, v4);
    v27 = *(_DWORD *)(v3 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(v3 + 8);
      v30 = 0;
      v31 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v32 = 1;
      v24 = *(_DWORD *)(v3 + 16) + 1;
      result = v38;
      v15 = (__int64 *)(v29 + 16LL * v31);
      v33 = *v15;
      if ( *v15 != a2 )
      {
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v30 )
            v30 = v15;
          v31 = v28 & (v32 + v31);
          v15 = (__int64 *)(v29 + 16LL * v31);
          v33 = *v15;
          if ( *v15 == a2 )
            goto LABEL_20;
          ++v32;
        }
        if ( v30 )
          v15 = v30;
      }
      goto LABEL_20;
    }
LABEL_58:
    ++*(_DWORD *)(v3 + 16);
    BUG();
  }
LABEL_20:
  *(_DWORD *)(v3 + 16) = v24;
  if ( *v15 != -4096 )
    --*(_DWORD *)(v3 + 20);
  *v15 = a2;
  *((_DWORD *)v15 + 2) = 0;
  *((_DWORD *)v15 + 2) = result;
  return result;
}
