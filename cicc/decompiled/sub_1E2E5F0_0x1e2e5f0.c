// Function: sub_1E2E5F0
// Address: 0x1e2e5f0
//
__int64 *__fastcall sub_1E2E5F0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int64 *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rax
  _DWORD *v11; // rcx
  __int64 *result; // rax
  int v13; // r11d
  __int64 *v14; // r9
  int v15; // eax
  int v16; // edx
  _QWORD *v17; // r13
  __int64 v18; // rax
  _QWORD *v19; // r13
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rcx
  int v31; // eax
  int v32; // ecx
  __int64 v33; // rdi
  unsigned int v34; // eax
  __int64 v35; // rsi
  int v36; // r9d
  __int64 *v37; // r8
  int v38; // eax
  int v39; // eax
  __int64 v40; // rsi
  int v41; // r8d
  unsigned int v42; // r14d
  __int64 *v43; // rdi
  __int64 v44; // rcx
  __int64 *v45; // r10
  unsigned __int64 v46; // [rsp+0h] [rbp-30h]
  _QWORD v47[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = a1 + 8;
  v47[0] = a2;
  v5 = *(_DWORD *)(a1 + 32);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_42;
  }
  v6 = *(_QWORD *)(a1 + 16);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 32LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
  {
    v10 = v8[1];
    v11 = (_DWORD *)(v10 & 0xFFFFFFFFFFFFFFF8LL);
    goto LABEL_4;
  }
  v13 = 1;
  v14 = 0;
  while ( 1 )
  {
    if ( v9 == -8 )
    {
      v15 = *(_DWORD *)(a1 + 24);
      if ( v14 )
        v8 = v14;
      ++*(_QWORD *)(a1 + 8);
      v16 = v15 + 1;
      if ( 4 * (v15 + 1) < 3 * v5 )
      {
        if ( v5 - *(_DWORD *)(a1 + 28) - v16 > v5 >> 3 )
        {
LABEL_14:
          *(_DWORD *)(a1 + 24) = v16;
          if ( *v8 != -8 )
            --*(_DWORD *)(a1 + 28);
          *v8 = a2;
          v8[1] = 0;
          v8[2] = 0;
          *((_DWORD *)v8 + 6) = 0;
          goto LABEL_17;
        }
        sub_1E2E3C0(v4, v5);
        v38 = *(_DWORD *)(a1 + 32);
        if ( v38 )
        {
          v39 = v38 - 1;
          v40 = *(_QWORD *)(a1 + 16);
          v41 = 1;
          v42 = v39 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v16 = *(_DWORD *)(a1 + 24) + 1;
          v43 = 0;
          v8 = (__int64 *)(v40 + 32LL * v42);
          v44 = *v8;
          if ( a2 != *v8 )
          {
            while ( v44 != -8 )
            {
              if ( v44 == -16 && !v43 )
                v43 = v8;
              v42 = v39 & (v41 + v42);
              v8 = (__int64 *)(v40 + 32LL * v42);
              v44 = *v8;
              if ( a2 == *v8 )
                goto LABEL_14;
              ++v41;
            }
            if ( v43 )
              v8 = v43;
          }
          goto LABEL_14;
        }
LABEL_71:
        ++*(_DWORD *)(a1 + 24);
        BUG();
      }
LABEL_42:
      sub_1E2E3C0(v4, 2 * v5);
      v31 = *(_DWORD *)(a1 + 32);
      if ( v31 )
      {
        v32 = v31 - 1;
        v33 = *(_QWORD *)(a1 + 16);
        v34 = (v31 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v16 = *(_DWORD *)(a1 + 24) + 1;
        v8 = (__int64 *)(v33 + 32LL * v34);
        v35 = *v8;
        if ( a2 != *v8 )
        {
          v36 = 1;
          v37 = 0;
          while ( v35 != -8 )
          {
            if ( !v37 && v35 == -16 )
              v37 = v8;
            v34 = v32 & (v36 + v34);
            v8 = (__int64 *)(v33 + 32LL * v34);
            v35 = *v8;
            if ( a2 == *v8 )
              goto LABEL_14;
            ++v36;
          }
          if ( v37 )
            v8 = v37;
        }
        goto LABEL_14;
      }
      goto LABEL_71;
    }
    if ( v9 != -16 || v14 )
      v8 = v14;
    v7 = (v5 - 1) & (v13 + v7);
    v45 = (__int64 *)(v6 + 32LL * v7);
    v9 = *v45;
    if ( a2 == *v45 )
      break;
    ++v13;
    v14 = v8;
    v8 = (__int64 *)(v6 + 32LL * v7);
  }
  v10 = v45[1];
  v8 = (__int64 *)(v6 + 32LL * v7);
  v11 = (_DWORD *)(v10 & 0xFFFFFFFFFFFFFFF8LL);
LABEL_4:
  if ( !v11 )
    goto LABEL_17;
  if ( (v10 & 4) == 0 )
    return v8 + 1;
  if ( v11[2] )
    return *(__int64 **)v11;
LABEL_17:
  v17 = *(_QWORD **)(a1 + 48);
  if ( v17 == *(_QWORD **)(a1 + 56) )
  {
    sub_1E2D160((char **)(a1 + 40), *(char **)(a1 + 48), v47);
    v19 = *(_QWORD **)(a1 + 48);
  }
  else
  {
    if ( v17 )
    {
      v18 = v47[0];
      v17[1] = 2;
      v17[2] = 0;
      v17[3] = v18;
      if ( v18 != -8 && v18 != 0 && v18 != -16 )
        sub_164C220((__int64)(v17 + 1));
      v17[4] = 0;
      *v17 = &unk_49FBDD8;
      v17 = *(_QWORD **)(a1 + 48);
    }
    v19 = v17 + 5;
    *(_QWORD *)(a1 + 48) = v19;
  }
  *(v19 - 1) = a1;
  *((_DWORD *)v8 + 6) = -858993459 * ((__int64)(*(_QWORD *)(a1 + 48) - *(_QWORD *)(a1 + 40)) >> 3) - 1;
  v8[2] = *(_QWORD *)(v47[0] + 56LL);
  v22 = sub_38BFA60(*(_QWORD *)a1, 1);
  v23 = v8[1];
  v24 = v23 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v23 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (v23 & 4) == 0 )
    {
      v25 = sub_22077B0(48);
      if ( v25 )
      {
        *(_QWORD *)v25 = v25 + 16;
        *(_QWORD *)(v25 + 8) = 0x400000000LL;
      }
      v26 = v25;
      v27 = v25 & 0xFFFFFFFFFFFFFFF8LL;
      v8[1] = v26 | 4;
      v28 = *(unsigned int *)(v27 + 8);
      if ( (unsigned int)v28 >= *(_DWORD *)(v27 + 12) )
      {
        v46 = v27;
        sub_16CD150(v27, (const void *)(v27 + 16), 0, 8, v20, v21);
        v27 = v46;
        v28 = *(unsigned int *)(v46 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v27 + 8 * v28) = v24;
      ++*(_DWORD *)(v27 + 8);
      v24 = v8[1] & 0xFFFFFFFFFFFFFFF8LL;
    }
    v29 = *(unsigned int *)(v24 + 8);
    if ( (unsigned int)v29 >= *(_DWORD *)(v24 + 12) )
    {
      sub_16CD150(v24, (const void *)(v24 + 16), 0, 8, v20, v21);
      v29 = *(unsigned int *)(v24 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v24 + 8 * v29) = v22;
    ++*(_DWORD *)(v24 + 8);
    v30 = v8[1];
  }
  else
  {
    v8[1] = v22;
    v30 = v22;
  }
  result = 0;
  if ( (v30 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (v30 & 4) == 0 )
      return v8 + 1;
    return *(__int64 **)(v30 & 0xFFFFFFFFFFFFFFF8LL);
  }
  return result;
}
