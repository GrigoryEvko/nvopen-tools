// Function: sub_B85AD0
// Address: 0xb85ad0
//
__int64 __fastcall sub_B85AD0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rsi
  int v6; // r14d
  __int64 v7; // r8
  __int64 *v8; // rdx
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r10
  __int64 *v12; // rbx
  __int64 result; // rax
  int v14; // eax
  int v15; // ecx
  pthread_rwlock_t *v16; // rax
  int v17; // eax
  __int64 v18; // r8
  unsigned int v19; // eax
  int v20; // r10d
  __int64 *v21; // r9
  int v22; // eax
  int v23; // eax
  __int64 *v24; // r8
  unsigned int v25; // r13d
  int v26; // r9d

  v4 = a1 + 688;
  v5 = *(unsigned int *)(a1 + 712);
  if ( !(_DWORD)v5 )
  {
    ++*(_QWORD *)(a1 + 688);
    goto LABEL_20;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 696);
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( *v10 != a2 )
  {
    while ( v11 != -4096 )
    {
      if ( !v8 && v11 == -8192 )
        v8 = v10;
      v9 = (v5 - 1) & (v6 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( *v10 == a2 )
        goto LABEL_3;
      ++v6;
    }
    if ( !v8 )
      v8 = v10;
    v14 = *(_DWORD *)(a1 + 704);
    ++*(_QWORD *)(a1 + 688);
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < (unsigned int)(3 * v5) )
    {
      if ( (int)v5 - *(_DWORD *)(a1 + 708) - v15 > (unsigned int)v5 >> 3 )
      {
LABEL_15:
        *(_DWORD *)(a1 + 704) = v15;
        if ( *v8 != -4096 )
          --*(_DWORD *)(a1 + 708);
        *v8 = a2;
        v12 = v8 + 1;
        v8[1] = 0;
        goto LABEL_18;
      }
      sub_B858F0(v4, v5);
      v22 = *(_DWORD *)(a1 + 712);
      if ( v22 )
      {
        v23 = v22 - 1;
        v4 = *(_QWORD *)(a1 + 696);
        v24 = 0;
        v25 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v26 = 1;
        v15 = *(_DWORD *)(a1 + 704) + 1;
        v8 = (__int64 *)(v4 + 16LL * v25);
        v5 = *v8;
        if ( *v8 != a2 )
        {
          while ( v5 != -4096 )
          {
            if ( !v24 && v5 == -8192 )
              v24 = v8;
            v25 = v23 & (v26 + v25);
            v8 = (__int64 *)(v4 + 16LL * v25);
            v5 = *v8;
            if ( *v8 == a2 )
              goto LABEL_15;
            ++v26;
          }
          if ( v24 )
            v8 = v24;
        }
        goto LABEL_15;
      }
LABEL_43:
      ++*(_DWORD *)(a1 + 704);
      BUG();
    }
LABEL_20:
    sub_B858F0(v4, 2 * v5);
    v17 = *(_DWORD *)(a1 + 712);
    if ( v17 )
    {
      v5 = (unsigned int)(v17 - 1);
      v18 = *(_QWORD *)(a1 + 696);
      v19 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v15 = *(_DWORD *)(a1 + 704) + 1;
      v8 = (__int64 *)(v18 + 16LL * v19);
      v4 = *v8;
      if ( *v8 != a2 )
      {
        v20 = 1;
        v21 = 0;
        while ( v4 != -4096 )
        {
          if ( !v21 && v4 == -8192 )
            v21 = v8;
          v19 = v5 & (v20 + v19);
          v8 = (__int64 *)(v18 + 16LL * v19);
          v4 = *v8;
          if ( *v8 == a2 )
            goto LABEL_15;
          ++v20;
        }
        if ( v21 )
          v8 = v21;
      }
      goto LABEL_15;
    }
    goto LABEL_43;
  }
LABEL_3:
  v12 = v10 + 1;
  result = v10[1];
  if ( !result )
  {
LABEL_18:
    v16 = (pthread_rwlock_t *)sub_BC2B00(v4, v5);
    result = sub_BC2C30(v16);
    *v12 = result;
  }
  return result;
}
