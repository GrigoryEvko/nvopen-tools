// Function: sub_DB0AC0
// Address: 0xdb0ac0
//
__int64 *__fastcall sub_DB0AC0(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rdi
  unsigned int v8; // esi
  __int64 v9; // rcx
  int v10; // r9d
  __int64 *v11; // r8
  unsigned int v12; // edx
  __int64 *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdi
  int v18; // eax
  int v19; // edx
  int v20; // eax
  int v21; // eax
  int v22; // ecx
  __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // rsi
  int v26; // r9d
  __int64 *v27; // r8
  int v28; // eax
  int v29; // eax
  __int64 v30; // rsi
  int v31; // r8d
  __int64 *v32; // rdi
  unsigned int v33; // r15d
  __int64 v34; // rcx

  v6 = a1 + 968;
  v7 = a1 + 1000;
  if ( a3 )
    v6 = v7;
  v8 = *(_DWORD *)(v6 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)v6;
    goto LABEL_27;
  }
  v9 = *(_QWORD *)(v6 + 8);
  v10 = 1;
  v11 = 0;
  v12 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__int64 *)(v9 + 40LL * v12);
  v14 = *v13;
  if ( *v13 == a2 )
  {
LABEL_5:
    if ( *((_DWORD *)v13 + 4) > 0x40u )
    {
      v15 = v13[1];
      if ( v15 )
        j_j___libc_free_0_0(v15);
    }
    v13[1] = *(_QWORD *)a4;
    *((_DWORD *)v13 + 4) = *(_DWORD *)(a4 + 8);
    *(_DWORD *)(a4 + 8) = 0;
    if ( *((_DWORD *)v13 + 8) > 0x40u )
    {
      v16 = v13[3];
      if ( v16 )
        j_j___libc_free_0_0(v16);
    }
    v13[3] = *(_QWORD *)(a4 + 16);
    *((_DWORD *)v13 + 8) = *(_DWORD *)(a4 + 24);
    *(_DWORD *)(a4 + 24) = 0;
    return v13 + 1;
  }
  while ( v14 != -4096 )
  {
    if ( !v11 && v14 == -8192 )
      v11 = v13;
    v12 = (v8 - 1) & (v10 + v12);
    v13 = (__int64 *)(v9 + 40LL * v12);
    v14 = *v13;
    if ( *v13 == a2 )
      goto LABEL_5;
    ++v10;
  }
  v18 = *(_DWORD *)(v6 + 16);
  if ( v11 )
    v13 = v11;
  ++*(_QWORD *)v6;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v8 )
  {
LABEL_27:
    sub_DB0860(v6, 2 * v8);
    v21 = *(_DWORD *)(v6 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(v6 + 8);
      v24 = (v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (__int64 *)(v23 + 40LL * v24);
      v25 = *v13;
      v19 = *(_DWORD *)(v6 + 16) + 1;
      if ( *v13 != a2 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -4096 )
        {
          if ( !v27 && v25 == -8192 )
            v27 = v13;
          v24 = v22 & (v26 + v24);
          v13 = (__int64 *)(v23 + 40LL * v24);
          v25 = *v13;
          if ( *v13 == a2 )
            goto LABEL_23;
          ++v26;
        }
        if ( v27 )
          v13 = v27;
      }
      goto LABEL_23;
    }
    goto LABEL_50;
  }
  if ( v8 - *(_DWORD *)(v6 + 20) - v19 <= v8 >> 3 )
  {
    sub_DB0860(v6, v8);
    v28 = *(_DWORD *)(v6 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(v6 + 8);
      v31 = 1;
      v32 = 0;
      v33 = v29 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (__int64 *)(v30 + 40LL * v33);
      v34 = *v13;
      v19 = *(_DWORD *)(v6 + 16) + 1;
      if ( *v13 != a2 )
      {
        while ( v34 != -4096 )
        {
          if ( v34 == -8192 && !v32 )
            v32 = v13;
          v33 = v29 & (v31 + v33);
          v13 = (__int64 *)(v30 + 40LL * v33);
          v34 = *v13;
          if ( *v13 == a2 )
            goto LABEL_23;
          ++v31;
        }
        if ( v32 )
          v13 = v32;
      }
      goto LABEL_23;
    }
LABEL_50:
    ++*(_DWORD *)(v6 + 16);
    BUG();
  }
LABEL_23:
  *(_DWORD *)(v6 + 16) = v19;
  if ( *v13 != -4096 )
    --*(_DWORD *)(v6 + 20);
  *v13 = a2;
  v20 = *(_DWORD *)(a4 + 8);
  *(_DWORD *)(a4 + 8) = 0;
  *((_DWORD *)v13 + 4) = v20;
  v13[1] = *(_QWORD *)a4;
  *((_DWORD *)v13 + 8) = *(_DWORD *)(a4 + 24);
  v13[3] = *(_QWORD *)(a4 + 16);
  *(_DWORD *)(a4 + 24) = 0;
  return v13 + 1;
}
