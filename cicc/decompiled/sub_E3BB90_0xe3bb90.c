// Function: sub_E3BB90
// Address: 0xe3bb90
//
unsigned __int64 __fastcall sub_E3BB90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned int v6; // esi
  __int64 v7; // r8
  unsigned int v8; // edi
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 *v11; // r15
  unsigned int v12; // esi
  __int64 v13; // r8
  unsigned int v14; // edi
  unsigned __int64 result; // rax
  __int64 v16; // rcx
  int v17; // r11d
  __int64 *v18; // rdx
  int v19; // eax
  int v20; // ecx
  int v21; // r11d
  __int64 *v22; // rdx
  int v23; // eax
  int v24; // ecx
  int v25; // eax
  int v26; // r8d
  __int64 v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // rsi
  int v30; // r10d
  __int64 *v31; // r9
  int v32; // eax
  int v33; // edi
  __int64 v34; // r8
  __int64 v35; // rsi
  int v36; // r10d
  __int64 *v37; // r9
  int v38; // eax
  int v39; // esi
  __int64 v40; // r8
  __int64 *v41; // rdi
  unsigned int v42; // r14d
  int v43; // r9d
  int v44; // eax
  int v45; // esi
  __int64 v46; // rdi
  __int64 *v47; // r8
  unsigned int v48; // r15d
  int v49; // r9d
  __int64 v50; // rax
  __int64 v51[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = a3;
  v51[0] = a2;
  sub_E3B670(a3 + 56, v51);
  *(_DWORD *)(v4 + 184) = 0;
  v6 = *(_DWORD *)(a1 + 32);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_27;
  }
  v7 = *(_QWORD *)(a1 + 16);
  v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( *v9 == a2 )
    goto LABEL_3;
  v17 = 1;
  v18 = 0;
  while ( v10 != -4096 )
  {
    if ( v18 || v10 != -8192 )
      v9 = v18;
    v8 = (v6 - 1) & (v17 + v8);
    v10 = *(_QWORD *)(v7 + 16LL * v8);
    if ( v10 == a2 )
      goto LABEL_3;
    ++v17;
    v18 = v9;
    v9 = (__int64 *)(v7 + 16LL * v8);
  }
  if ( !v18 )
    v18 = v9;
  v19 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v6 )
  {
LABEL_27:
    sub_E39660(a1 + 8, 2 * v6);
    v25 = *(_DWORD *)(a1 + 32);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 16);
      v28 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v20 = *(_DWORD *)(a1 + 24) + 1;
      v18 = (__int64 *)(v27 + 16LL * v28);
      v29 = *v18;
      if ( *v18 != a2 )
      {
        v30 = 1;
        v31 = 0;
        while ( v29 != -4096 )
        {
          if ( v29 == -8192 && !v31 )
            v31 = v18;
          v28 = v26 & (v30 + v28);
          v18 = (__int64 *)(v27 + 16LL * v28);
          v29 = *v18;
          if ( *v18 == a2 )
            goto LABEL_14;
          ++v30;
        }
        if ( v31 )
          v18 = v31;
      }
      goto LABEL_14;
    }
    goto LABEL_84;
  }
  if ( v6 - *(_DWORD *)(a1 + 28) - v20 <= v6 >> 3 )
  {
    sub_E39660(a1 + 8, v6);
    v44 = *(_DWORD *)(a1 + 32);
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 16);
      v47 = 0;
      v48 = (v44 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v49 = 1;
      v20 = *(_DWORD *)(a1 + 24) + 1;
      v18 = (__int64 *)(v46 + 16LL * v48);
      v50 = *v18;
      if ( *v18 != a2 )
      {
        while ( v50 != -4096 )
        {
          if ( v50 == -8192 && !v47 )
            v47 = v18;
          v48 = v45 & (v49 + v48);
          v18 = (__int64 *)(v46 + 16LL * v48);
          v50 = *v18;
          if ( *v18 == a2 )
            goto LABEL_14;
          ++v49;
        }
        if ( v47 )
          v18 = v47;
      }
      goto LABEL_14;
    }
LABEL_84:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 24) = v20;
  if ( *v18 != -4096 )
    --*(_DWORD *)(a1 + 28);
  *v18 = a2;
  v18[1] = v4;
LABEL_3:
  v11 = *(__int64 **)v4;
  if ( *(_QWORD *)v4 )
  {
    do
    {
      v51[0] = a2;
      v4 = (__int64)v11;
      sub_E3B670((__int64)(v11 + 7), v51);
      *((_DWORD *)v11 + 46) = 0;
      v11 = (__int64 *)*v11;
    }
    while ( v11 );
  }
  v12 = *(_DWORD *)(a1 + 64);
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_35;
  }
  v13 = *(_QWORD *)(a1 + 48);
  v14 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v13 + 16LL * v14;
  v16 = *(_QWORD *)result;
  if ( *(_QWORD *)result == a2 )
    goto LABEL_7;
  v21 = 1;
  v22 = 0;
  while ( v16 != -4096 )
  {
    if ( v22 || v16 != -8192 )
      result = (unsigned __int64)v22;
    v14 = (v12 - 1) & (v21 + v14);
    v16 = *(_QWORD *)(v13 + 16LL * v14);
    if ( v16 == a2 )
      goto LABEL_7;
    ++v21;
    v22 = (__int64 *)result;
    result = v13 + 16LL * v14;
  }
  if ( !v22 )
    v22 = (__int64 *)result;
  v23 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  v24 = v23 + 1;
  if ( 4 * (v23 + 1) >= 3 * v12 )
  {
LABEL_35:
    sub_E39120(a1 + 40, 2 * v12);
    v32 = *(_DWORD *)(a1 + 64);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 48);
      result = (v32 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v24 = *(_DWORD *)(a1 + 56) + 1;
      v22 = (__int64 *)(v34 + 16 * result);
      v35 = *v22;
      if ( *v22 != a2 )
      {
        v36 = 1;
        v37 = 0;
        while ( v35 != -4096 )
        {
          if ( v35 == -8192 && !v37 )
            v37 = v22;
          result = v33 & (unsigned int)(v36 + result);
          v22 = (__int64 *)(v34 + 16LL * (unsigned int)result);
          v35 = *v22;
          if ( *v22 == a2 )
            goto LABEL_23;
          ++v36;
        }
        if ( v37 )
          v22 = v37;
      }
      goto LABEL_23;
    }
    goto LABEL_85;
  }
  result = v12 - *(_DWORD *)(a1 + 60) - v24;
  if ( (unsigned int)result <= v12 >> 3 )
  {
    sub_E39120(a1 + 40, v12);
    v38 = *(_DWORD *)(a1 + 64);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a1 + 48);
      v41 = 0;
      v42 = (v38 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v43 = 1;
      v24 = *(_DWORD *)(a1 + 56) + 1;
      v22 = (__int64 *)(v40 + 16LL * v42);
      result = *v22;
      if ( *v22 != a2 )
      {
        while ( result != -4096 )
        {
          if ( !v41 && result == -8192 )
            v41 = v22;
          v42 = v39 & (v43 + v42);
          v22 = (__int64 *)(v40 + 16LL * v42);
          result = *v22;
          if ( *v22 == a2 )
            goto LABEL_23;
          ++v43;
        }
        if ( v41 )
          v22 = v41;
      }
      goto LABEL_23;
    }
LABEL_85:
    ++*(_DWORD *)(a1 + 56);
    BUG();
  }
LABEL_23:
  *(_DWORD *)(a1 + 56) = v24;
  if ( *v22 != -4096 )
    --*(_DWORD *)(a1 + 60);
  *v22 = a2;
  v22[1] = v4;
LABEL_7:
  *(_DWORD *)(v4 + 184) = 0;
  return result;
}
