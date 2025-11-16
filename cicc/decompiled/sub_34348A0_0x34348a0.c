// Function: sub_34348A0
// Address: 0x34348a0
//
__int64 *__fastcall sub_34348A0(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v5; // r10
  int v6; // r11d
  int v7; // r8d
  __int64 v8; // rax
  unsigned int i; // edx
  __int64 *v10; // rcx
  __int64 v11; // r13
  unsigned int v12; // edx
  __int64 *result; // rax
  int v14; // r14d
  int v15; // eax
  int v16; // ecx
  int v17; // edi
  __int64 v18; // r10
  __int64 v19; // r8
  int v20; // r11d
  unsigned int k; // edx
  unsigned int v22; // edx
  int v23; // ecx
  int v24; // edx
  int v25; // ecx
  int v26; // eax
  int v27; // ecx
  int v28; // edi
  int v29; // r10d
  __int64 v30; // r8
  unsigned int j; // edx
  unsigned int v32; // edx
  int v33; // r9d
  int v34; // r9d

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_14;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v7 = *(_DWORD *)(a2 + 8);
  v8 = 0;
  for ( i = (v3 - 1) & (v7 + ((*(_QWORD *)a2 >> 9) ^ (*(_QWORD *)a2 >> 4))); ; i = (v3 - 1) & v12 )
  {
    v10 = (__int64 *)(v5 + 32LL * i);
    v11 = *v10;
    if ( *(_QWORD *)a2 == *v10 && v7 == *((_DWORD *)v10 + 2) )
      return v10 + 2;
    if ( !v11 )
      break;
LABEL_5:
    v12 = v6 + i;
    ++v6;
  }
  v14 = *((_DWORD *)v10 + 2);
  if ( v14 != -1 )
  {
    if ( v14 == -2 && !v8 )
      v8 = v5 + 32LL * i;
    goto LABEL_5;
  }
  if ( !v8 )
    v8 = v5 + 32LL * i;
  v25 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v23 = v25 + 1;
  if ( 4 * v23 < 3 * v3 )
  {
    if ( v3 - *(_DWORD *)(a1 + 20) - v23 <= v3 >> 3 )
    {
      sub_3434670(a1, v3);
      v26 = *(_DWORD *)(a1 + 24);
      if ( v26 )
      {
        v27 = v26 - 1;
        v28 = *(_DWORD *)(a2 + 8);
        v29 = 1;
        for ( j = (v26 - 1) & (v28 + ((*(_QWORD *)a2 >> 9) ^ (*(_QWORD *)a2 >> 4))); ; j = v27 & v32 )
        {
          v30 = *(_QWORD *)(a1 + 8);
          v8 = v30 + 32LL * j;
          if ( *(_QWORD *)a2 == *(_QWORD *)v8 && v28 == *(_DWORD *)(v8 + 8) )
            break;
          if ( !*(_QWORD *)v8 )
          {
            v34 = *(_DWORD *)(v8 + 8);
            if ( v34 == -1 )
            {
              v23 = *(_DWORD *)(a1 + 16) + 1;
              if ( v11 )
                v8 = v11;
              goto LABEL_21;
            }
            if ( v34 == -2 && !v11 )
              v11 = v30 + 32LL * j;
          }
          v32 = v29 + j;
          ++v29;
        }
        goto LABEL_20;
      }
LABEL_53:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
    goto LABEL_21;
  }
LABEL_14:
  sub_3434670(a1, 2 * v3);
  v15 = *(_DWORD *)(a1 + 24);
  if ( !v15 )
    goto LABEL_53;
  v16 = v15 - 1;
  v17 = *(_DWORD *)(a2 + 8);
  v18 = 0;
  v20 = 1;
  for ( k = (v15 - 1) & (v17 + ((*(_QWORD *)a2 >> 9) ^ (*(_QWORD *)a2 >> 4))); ; k = v16 & v22 )
  {
    v19 = *(_QWORD *)(a1 + 8);
    v8 = v19 + 32LL * k;
    if ( *(_QWORD *)a2 == *(_QWORD *)v8 && v17 == *(_DWORD *)(v8 + 8) )
      break;
    if ( !*(_QWORD *)v8 )
    {
      v33 = *(_DWORD *)(v8 + 8);
      if ( v33 == -1 )
      {
        v23 = *(_DWORD *)(a1 + 16) + 1;
        if ( v18 )
          v8 = v18;
        goto LABEL_21;
      }
      if ( v33 == -2 && !v18 )
        v18 = v19 + 32LL * k;
    }
    v22 = v20 + k;
    ++v20;
  }
LABEL_20:
  v23 = *(_DWORD *)(a1 + 16) + 1;
LABEL_21:
  *(_DWORD *)(a1 + 16) = v23;
  if ( *(_QWORD *)v8 || *(_DWORD *)(v8 + 8) != -1 )
    --*(_DWORD *)(a1 + 20);
  result = (__int64 *)(v8 + 16);
  *(result - 2) = *(_QWORD *)a2;
  v24 = *(_DWORD *)(a2 + 8);
  *result = 0;
  *((_DWORD *)result - 2) = v24;
  *((_DWORD *)result + 2) = 0;
  return result;
}
