// Function: sub_2E7DA40
// Address: 0x2e7da40
//
__int64 __fastcall sub_2E7DA40(__int64 a1, __int64 a2, const void *a3, __int64 a4)
{
  __int64 v4; // r10
  unsigned int v9; // esi
  __int64 v10; // r9
  int v11; // r11d
  __int64 v12; // r8
  unsigned int v13; // edi
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // rcx
  __int64 result; // rax
  unsigned __int64 v18; // rcx
  __int64 v19; // r13
  signed __int64 v20; // r12
  unsigned __int64 v21; // rdx
  int v22; // ecx
  int v23; // ecx
  int v24; // eax
  int v25; // esi
  unsigned int v26; // edx
  __int64 v27; // rdi
  int v28; // r10d
  int v29; // eax
  int v30; // edx
  __int64 v31; // rdi
  unsigned int v32; // r15d
  __int64 v33; // rsi

  v4 = a1 + 456;
  v9 = *(_DWORD *)(a1 + 480);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 456);
    goto LABEL_23;
  }
  v10 = v9 - 1;
  v11 = 1;
  v12 = *(_QWORD *)(a1 + 464);
  v13 = v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = v12 + 40LL * v13;
  v15 = 0;
  v16 = *(_QWORD *)v14;
  if ( *(_QWORD *)v14 == a2 )
  {
LABEL_3:
    result = *(unsigned int *)(v14 + 16);
    v18 = *(unsigned int *)(v14 + 20);
    v19 = v14 + 8;
    goto LABEL_4;
  }
  while ( v16 != -4096 )
  {
    if ( !v15 && v16 == -8192 )
      v15 = (__int64 *)v14;
    v13 = v10 & (v11 + v13);
    v14 = v12 + 40LL * v13;
    v16 = *(_QWORD *)v14;
    if ( *(_QWORD *)v14 == a2 )
      goto LABEL_3;
    ++v11;
  }
  v22 = *(_DWORD *)(a1 + 472);
  if ( !v15 )
    v15 = (__int64 *)v14;
  ++*(_QWORD *)(a1 + 456);
  v23 = v22 + 1;
  if ( 4 * v23 >= 3 * v9 )
  {
LABEL_23:
    sub_2E7D720(v4, 2 * v9);
    v24 = *(_DWORD *)(a1 + 480);
    if ( v24 )
    {
      v25 = v24 - 1;
      v12 = *(_QWORD *)(a1 + 464);
      v26 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = *(_DWORD *)(a1 + 472) + 1;
      v15 = (__int64 *)(v12 + 40LL * v26);
      v27 = *v15;
      if ( *v15 != a2 )
      {
        v28 = 1;
        v10 = 0;
        while ( v27 != -4096 )
        {
          if ( !v10 && v27 == -8192 )
            v10 = (__int64)v15;
          v26 = v25 & (v28 + v26);
          v15 = (__int64 *)(v12 + 40LL * v26);
          v27 = *v15;
          if ( *v15 == a2 )
            goto LABEL_19;
          ++v28;
        }
        if ( v10 )
          v15 = (__int64 *)v10;
      }
      goto LABEL_19;
    }
    goto LABEL_46;
  }
  if ( v9 - *(_DWORD *)(a1 + 476) - v23 <= v9 >> 3 )
  {
    sub_2E7D720(v4, v9);
    v29 = *(_DWORD *)(a1 + 480);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 464);
      v12 = 0;
      v32 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = 1;
      v23 = *(_DWORD *)(a1 + 472) + 1;
      v15 = (__int64 *)(v31 + 40LL * v32);
      v33 = *v15;
      if ( *v15 != a2 )
      {
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v12 )
            v12 = (__int64)v15;
          v32 = v30 & (v10 + v32);
          v15 = (__int64 *)(v31 + 40LL * v32);
          v33 = *v15;
          if ( *v15 == a2 )
            goto LABEL_19;
          v10 = (unsigned int)(v10 + 1);
        }
        if ( v12 )
          v15 = (__int64 *)v12;
      }
      goto LABEL_19;
    }
LABEL_46:
    ++*(_DWORD *)(a1 + 472);
    BUG();
  }
LABEL_19:
  *(_DWORD *)(a1 + 472) = v23;
  if ( *v15 != -4096 )
    --*(_DWORD *)(a1 + 476);
  *v15 = a2;
  v19 = (__int64)(v15 + 1);
  v15[2] = 0x400000000LL;
  v18 = 4;
  v15[1] = (__int64)(v15 + 3);
  result = 0;
LABEL_4:
  v20 = 4 * a4;
  v21 = (v20 >> 2) + result;
  if ( v18 < v21 )
  {
    sub_C8D5F0(v19, (const void *)(v19 + 16), v21, 4u, v12, v10);
    result = *(unsigned int *)(v19 + 8);
  }
  if ( v20 )
  {
    memcpy((void *)(*(_QWORD *)v19 + 4 * result), a3, v20);
    result = *(unsigned int *)(v19 + 8);
  }
  *(_DWORD *)(v19 + 8) = result + (v20 >> 2);
  return result;
}
