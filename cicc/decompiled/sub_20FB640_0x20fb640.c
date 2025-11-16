// Function: sub_20FB640
// Address: 0x20fb640
//
void __fastcall sub_20FB640(_QWORD *a1, __int64 a2, __int64 a3)
{
  void *v5; // rdi
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 *v13; // r8
  __int64 *v14; // r9
  __int64 v15; // rsi
  __int64 *v16; // rdi
  unsigned int v17; // r10d
  __int64 *v18; // rax
  __int64 *v19; // rcx
  __int64 v20; // r12
  __int64 v21; // r13
  __int64 v22; // r12
  __int64 *v23; // rdi
  __int64 *v24; // r8
  __int64 *v25; // rsi
  unsigned int v26; // r9d
  __int64 *v27; // rax
  __int64 *v28; // rcx

  ++*(_QWORD *)a3;
  v5 = *(void **)(a3 + 16);
  if ( v5 != *(void **)(a3 + 8) )
  {
    v6 = 4 * (*(_DWORD *)(a3 + 28) - *(_DWORD *)(a3 + 32));
    v7 = *(unsigned int *)(a3 + 24);
    if ( v6 < 0x20 )
      v6 = 32;
    if ( (unsigned int)v7 > v6 )
    {
      sub_16CC920(a3);
      goto LABEL_7;
    }
    memset(v5, -1, 8 * v7);
  }
  *(_QWORD *)(a3 + 28) = 0;
LABEL_7:
  if ( !a2 )
    return;
  v8 = *(unsigned int *)(a2 + 8);
  v9 = 0;
  if ( (_DWORD)v8 == 2 )
    v9 = *(_QWORD *)(a2 - 8);
  v10 = sub_20FB390(a1, *(unsigned __int8 **)(a2 - 8 * v8), v9);
  if ( !v10 )
    return;
  if ( a1[28] == v10 )
  {
    v20 = *a1;
    v21 = *(_QWORD *)(*a1 + 328LL);
    v22 = v20 + 320;
    if ( v22 == v21 )
      return;
    v23 = *(__int64 **)(a3 + 16);
    v24 = *(__int64 **)(a3 + 8);
    while ( 1 )
    {
      if ( v23 != v24 )
        goto LABEL_30;
      v25 = &v23[*(unsigned int *)(a3 + 28)];
      v26 = *(_DWORD *)(a3 + 28);
      if ( v25 != v23 )
      {
        v27 = v23;
        v28 = 0;
        while ( v21 != *v27 )
        {
          if ( *v27 == -2 )
            v28 = v27;
          if ( v25 == ++v27 )
          {
            if ( !v28 )
              goto LABEL_42;
            *v28 = v21;
            v23 = *(__int64 **)(a3 + 16);
            --*(_DWORD *)(a3 + 32);
            v24 = *(__int64 **)(a3 + 8);
            ++*(_QWORD *)a3;
            goto LABEL_31;
          }
        }
        goto LABEL_31;
      }
LABEL_42:
      if ( v26 < *(_DWORD *)(a3 + 24) )
      {
        *(_DWORD *)(a3 + 28) = v26 + 1;
        *v25 = v21;
        v24 = *(__int64 **)(a3 + 8);
        ++*(_QWORD *)a3;
        v23 = *(__int64 **)(a3 + 16);
      }
      else
      {
LABEL_30:
        sub_16CCBA0(a3, v21);
        v23 = *(__int64 **)(a3 + 16);
        v24 = *(__int64 **)(a3 + 8);
      }
LABEL_31:
      v21 = *(_QWORD *)(v21 + 8);
      if ( v22 == v21 )
        return;
    }
  }
  v11 = *(_QWORD *)(v10 + 80);
  v12 = v11 + 16LL * *(unsigned int *)(v10 + 88);
  if ( v11 != v12 )
  {
    v13 = *(__int64 **)(a3 + 16);
    v14 = *(__int64 **)(a3 + 8);
    do
    {
LABEL_16:
      v15 = *(_QWORD *)(*(_QWORD *)v11 + 24LL);
      if ( v13 != v14 )
        goto LABEL_14;
      v16 = &v13[*(unsigned int *)(a3 + 28)];
      v17 = *(_DWORD *)(a3 + 28);
      if ( v16 != v13 )
      {
        v18 = v13;
        v19 = 0;
        while ( v15 != *v18 )
        {
          if ( *v18 == -2 )
            v19 = v18;
          if ( v16 == ++v18 )
          {
            if ( !v19 )
              goto LABEL_26;
            v11 += 16;
            *v19 = v15;
            v13 = *(__int64 **)(a3 + 16);
            --*(_DWORD *)(a3 + 32);
            v14 = *(__int64 **)(a3 + 8);
            ++*(_QWORD *)a3;
            if ( v12 != v11 )
              goto LABEL_16;
            return;
          }
        }
        goto LABEL_15;
      }
LABEL_26:
      if ( v17 < *(_DWORD *)(a3 + 24) )
      {
        *(_DWORD *)(a3 + 28) = v17 + 1;
        *v16 = v15;
        v14 = *(__int64 **)(a3 + 8);
        ++*(_QWORD *)a3;
        v13 = *(__int64 **)(a3 + 16);
      }
      else
      {
LABEL_14:
        sub_16CCBA0(a3, v15);
        v13 = *(__int64 **)(a3 + 16);
        v14 = *(__int64 **)(a3 + 8);
      }
LABEL_15:
      v11 += 16;
    }
    while ( v12 != v11 );
  }
}
