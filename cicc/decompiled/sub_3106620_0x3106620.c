// Function: sub_3106620
// Address: 0x3106620
//
__int64 __fastcall sub_3106620(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r12
  unsigned int v6; // esi
  __int64 v7; // rcx
  unsigned int v8; // edi
  unsigned int v9; // r8d
  unsigned __int64 *v10; // rdx
  unsigned __int64 v11; // rax
  unsigned int v12; // r8d
  unsigned __int64 *v13; // rdx
  unsigned __int64 v14; // rax
  __int64 result; // rax
  int v16; // r10d
  unsigned __int64 *v17; // r9
  int v18; // eax
  int v19; // edx
  int v20; // eax
  int v21; // ecx
  __int64 v22; // rdi
  unsigned int v23; // edx
  unsigned __int64 *v24; // r9
  unsigned __int64 v25; // rsi
  int v26; // eax
  int v27; // r10d
  unsigned __int64 *v28; // r8
  int v29; // r10d
  int v30; // eax
  int v31; // eax
  int v32; // ecx
  __int64 v33; // rdi
  unsigned int v34; // eax
  unsigned __int64 v35; // rsi
  int v36; // r10d
  unsigned __int64 *v37; // r8
  int v38; // eax
  int v39; // eax
  __int64 v40; // rsi
  int v41; // r8d
  unsigned int v42; // r15d
  unsigned __int64 *v43; // rdi
  unsigned __int64 v44; // rcx
  int v45; // eax
  int v46; // edx
  __int64 v47; // rsi
  int v48; // r8d
  unsigned __int64 *v49; // rdi
  unsigned int v50; // r12d
  unsigned __int64 v51; // rcx

  v3 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v4 = a2 | 4;
  *(_QWORD *)(a1 + 40) = a2;
  v6 = *(_DWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_37;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = v6 - 1;
  v9 = (v6 - 1) & (v4 ^ (v4 >> 9));
  v10 = (unsigned __int64 *)(v7 + 8LL * v9);
  v11 = *v10;
  if ( v4 == *v10 )
    goto LABEL_3;
  v16 = 1;
  v17 = 0;
  while ( v11 != -4 )
  {
    if ( v11 != -16 || v17 )
      v10 = v17;
    v9 = v8 & (v16 + v9);
    v11 = *(_QWORD *)(v7 + 8LL * v9);
    if ( v4 == v11 )
      goto LABEL_3;
    ++v16;
    v17 = v10;
    v10 = (unsigned __int64 *)(v7 + 8LL * v9);
  }
  v18 = *(_DWORD *)(a1 + 16);
  if ( !v17 )
    v17 = v10;
  ++*(_QWORD *)a1;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v6 )
  {
LABEL_37:
    sub_3106460(a1, 2 * v6);
    v31 = *(_DWORD *)(a1 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a1 + 8);
      v34 = (v31 - 1) & (v4 ^ (v4 >> 9));
      v17 = (unsigned __int64 *)(v33 + 8LL * v34);
      v35 = *v17;
      v19 = *(_DWORD *)(a1 + 16) + 1;
      if ( v4 != *v17 )
      {
        v36 = 1;
        v37 = 0;
        while ( v35 != -4 )
        {
          if ( v35 == -16 && !v37 )
            v37 = v17;
          v34 = v32 & (v36 + v34);
          v17 = (unsigned __int64 *)(v33 + 8LL * v34);
          v35 = *v17;
          if ( v4 == *v17 )
            goto LABEL_15;
          ++v36;
        }
        if ( v37 )
          v17 = v37;
      }
      goto LABEL_15;
    }
    goto LABEL_87;
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v19 <= v6 >> 3 )
  {
    sub_3106460(a1, v6);
    v38 = *(_DWORD *)(a1 + 24);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(a1 + 8);
      v41 = 1;
      v42 = v39 & (v4 ^ (v4 >> 9));
      v17 = (unsigned __int64 *)(v40 + 8LL * v42);
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v43 = 0;
      v44 = *v17;
      if ( v4 != *v17 )
      {
        while ( v44 != -4 )
        {
          if ( !v43 && v44 == -16 )
            v43 = v17;
          v42 = v39 & (v41 + v42);
          v17 = (unsigned __int64 *)(v40 + 8LL * v42);
          v44 = *v17;
          if ( v4 == *v17 )
            goto LABEL_15;
          ++v41;
        }
        if ( v43 )
          v17 = v43;
      }
      goto LABEL_15;
    }
LABEL_87:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *v17 != -4 )
    --*(_DWORD *)(a1 + 20);
  *v17 = v4;
  v6 = *(_DWORD *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 8);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_19;
  }
  v8 = v6 - 1;
LABEL_3:
  v12 = v8 & (v3 ^ (v3 >> 9));
  v13 = (unsigned __int64 *)(v7 + 8LL * v12);
  v14 = *v13;
  if ( v3 == *v13 )
    goto LABEL_4;
  v29 = 1;
  v24 = 0;
  while ( v14 != -4 )
  {
    if ( v14 != -16 || v24 )
      v13 = v24;
    v12 = v8 & (v29 + v12);
    v14 = *(_QWORD *)(v7 + 8LL * v12);
    if ( v3 == v14 )
      goto LABEL_4;
    ++v29;
    v24 = v13;
    v13 = (unsigned __int64 *)(v7 + 8LL * v12);
  }
  v30 = *(_DWORD *)(a1 + 16);
  if ( !v24 )
    v24 = v13;
  ++*(_QWORD *)a1;
  v26 = v30 + 1;
  if ( 4 * v26 >= 3 * v6 )
  {
LABEL_19:
    sub_3106460(a1, 2 * v6);
    v20 = *(_DWORD *)(a1 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 8);
      v23 = (v20 - 1) & (v3 ^ (v3 >> 9));
      v24 = (unsigned __int64 *)(v22 + 8LL * v23);
      v25 = *v24;
      v26 = *(_DWORD *)(a1 + 16) + 1;
      if ( v3 != *v24 )
      {
        v27 = 1;
        v28 = 0;
        while ( v25 != -4 )
        {
          if ( !v28 && v25 == -16 )
            v28 = v24;
          v23 = v21 & (v27 + v23);
          v24 = (unsigned __int64 *)(v22 + 8LL * v23);
          v25 = *v24;
          if ( v3 == *v24 )
            goto LABEL_32;
          ++v27;
        }
        if ( v28 )
          v24 = v28;
      }
      goto LABEL_32;
    }
    goto LABEL_86;
  }
  if ( v6 - (v26 + *(_DWORD *)(a1 + 20)) <= v6 >> 3 )
  {
    sub_3106460(a1, v6);
    v45 = *(_DWORD *)(a1 + 24);
    if ( v45 )
    {
      v46 = v45 - 1;
      v47 = *(_QWORD *)(a1 + 8);
      v48 = 1;
      v49 = 0;
      v50 = (v45 - 1) & (v3 ^ (v3 >> 9));
      v24 = (unsigned __int64 *)(v47 + 8LL * v50);
      v51 = *v24;
      v26 = *(_DWORD *)(a1 + 16) + 1;
      if ( v3 != *v24 )
      {
        while ( v51 != -4 )
        {
          if ( v51 == -16 && !v49 )
            v49 = v24;
          v50 = v46 & (v48 + v50);
          v24 = (unsigned __int64 *)(v47 + 8LL * v50);
          v51 = *v24;
          if ( v3 == *v24 )
            goto LABEL_32;
          ++v48;
        }
        if ( v49 )
          v24 = v49;
      }
      goto LABEL_32;
    }
LABEL_86:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_32:
  *(_DWORD *)(a1 + 16) = v26;
  if ( *v24 != -4 )
    --*(_DWORD *)(a1 + 20);
  *v24 = v3;
LABEL_4:
  result = *(_QWORD *)(a1 + 32);
  if ( *(_BYTE *)(result + 1) )
    *(_QWORD *)(a1 + 48) = a2;
  if ( *(_BYTE *)(result + 2) )
    *(_QWORD *)(a1 + 56) = a2;
  return result;
}
