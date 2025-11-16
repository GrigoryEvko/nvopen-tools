// Function: sub_DEEF40
// Address: 0xdeef40
//
__int64 __fastcall sub_DEEF40(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  unsigned int v4; // esi
  __int64 *v5; // r12
  __int64 v6; // r9
  __int64 v7; // r8
  int v8; // r10d
  unsigned int v9; // r13d
  unsigned int v10; // edi
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 result; // rax
  __int64 v15; // r13
  int v16; // edx
  int v17; // ecx
  int v18; // ecx
  int v19; // eax
  int v20; // esi
  unsigned int v21; // edx
  __int64 v22; // rdi
  int v23; // r10d
  int v24; // eax
  int v25; // edx
  __int64 v26; // rdi
  unsigned int v27; // r13d
  __int64 v28; // rsi

  v3 = sub_DD8400(*(_QWORD *)(a1 + 112), a2);
  v4 = *(_DWORD *)(a1 + 24);
  v5 = v3;
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_22;
  }
  v6 = v4 - 1;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = ((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4);
  v10 = v6 & v9;
  v11 = v7 + 24LL * ((unsigned int)v6 & v9);
  v12 = 0;
  v13 = *(_QWORD *)v11;
  if ( v5 != *(__int64 **)v11 )
  {
    while ( v13 != -4096 )
    {
      if ( !v12 && v13 == -8192 )
        v12 = v11;
      v10 = v6 & (v8 + v10);
      v11 = v7 + 24LL * v10;
      v13 = *(_QWORD *)v11;
      if ( v5 == *(__int64 **)v11 )
        goto LABEL_3;
      ++v8;
    }
    v17 = *(_DWORD *)(a1 + 16);
    if ( !v12 )
      v12 = v11;
    ++*(_QWORD *)a1;
    v18 = v17 + 1;
    if ( 4 * v18 < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 20) - v18 > v4 >> 3 )
      {
LABEL_17:
        *(_DWORD *)(a1 + 16) = v18;
        if ( *(_QWORD *)v12 != -4096 )
          --*(_DWORD *)(a1 + 20);
        *(_QWORD *)v12 = v5;
        v15 = v12 + 8;
        *(_DWORD *)(v12 + 8) = 0;
        *(_QWORD *)(v12 + 16) = 0;
        goto LABEL_20;
      }
      sub_DB02E0(a1, v4);
      v24 = *(_DWORD *)(a1 + 24);
      if ( v24 )
      {
        v25 = v24 - 1;
        v26 = *(_QWORD *)(a1 + 8);
        v6 = 1;
        v27 = (v24 - 1) & v9;
        v7 = 0;
        v18 = *(_DWORD *)(a1 + 16) + 1;
        v12 = v26 + 24LL * v27;
        v28 = *(_QWORD *)v12;
        if ( v5 != *(__int64 **)v12 )
        {
          while ( v28 != -4096 )
          {
            if ( !v7 && v28 == -8192 )
              v7 = v12;
            v27 = v25 & (v6 + v27);
            v12 = v26 + 24LL * v27;
            v28 = *(_QWORD *)v12;
            if ( v5 == *(__int64 **)v12 )
              goto LABEL_17;
            v6 = (unsigned int)(v6 + 1);
          }
          if ( v7 )
            v12 = v7;
        }
        goto LABEL_17;
      }
LABEL_45:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
LABEL_22:
    sub_DB02E0(a1, 2 * v4);
    v19 = *(_DWORD *)(a1 + 24);
    if ( v19 )
    {
      v20 = v19 - 1;
      v7 = *(_QWORD *)(a1 + 8);
      v21 = (v19 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v18 = *(_DWORD *)(a1 + 16) + 1;
      v12 = v7 + 24LL * v21;
      v22 = *(_QWORD *)v12;
      if ( v5 != *(__int64 **)v12 )
      {
        v23 = 1;
        v6 = 0;
        while ( v22 != -4096 )
        {
          if ( !v6 && v22 == -8192 )
            v6 = v12;
          v21 = v20 & (v23 + v21);
          v12 = v7 + 24LL * v21;
          v22 = *(_QWORD *)v12;
          if ( v5 == *(__int64 **)v12 )
            goto LABEL_17;
          ++v23;
        }
        if ( v6 )
          v12 = v6;
      }
      goto LABEL_17;
    }
    goto LABEL_45;
  }
LABEL_3:
  result = *(_QWORD *)(v11 + 16);
  v15 = v11 + 8;
  if ( result )
  {
    if ( *(_DWORD *)(a1 + 136) == *(_DWORD *)(v11 + 8) )
      return result;
    goto LABEL_5;
  }
LABEL_20:
  result = (__int64)v5;
LABEL_5:
  result = sub_DEEEC0(*(_QWORD *)(a1 + 112), result, *(_QWORD *)(a1 + 120), *(_QWORD *)(a1 + 128), v7, v6);
  v16 = *(_DWORD *)(a1 + 136);
  *(_QWORD *)(v15 + 8) = result;
  *(_DWORD *)v15 = v16;
  return result;
}
