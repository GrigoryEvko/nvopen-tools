// Function: sub_13848E0
// Address: 0x13848e0
//
__int64 __fastcall sub_13848E0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned int v8; // esi
  __int64 v9; // r8
  unsigned int v10; // edi
  _QWORD *v11; // r13
  __int64 v12; // rcx
  _QWORD *v13; // r8
  __int64 v14; // rdx
  __int64 v15; // r12
  unsigned __int64 v16; // rax
  unsigned int v17; // r8d
  int v19; // r10d
  _QWORD *v20; // rdx
  int v21; // ecx
  int v22; // edi
  unsigned __int64 v23; // rsi
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rsi
  unsigned int v27; // eax
  __int64 v28; // r8
  int v29; // r10d
  _QWORD *v30; // r9
  int v31; // edx
  int v32; // ecx
  __int64 v33; // r8
  int v34; // r10d
  unsigned int v35; // eax
  __int64 v36; // rsi
  _QWORD *v37; // r15
  _QWORD *v38; // rbx
  __int64 v39; // rdi
  _QWORD *v40; // [rsp+8h] [rbp-38h]
  _QWORD *v41; // [rsp+8h] [rbp-38h]
  _QWORD *v42; // [rsp+8h] [rbp-38h]

  v8 = *(_DWORD *)(a1 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_19;
  }
  v9 = *(_QWORD *)(a1 + 8);
  v10 = (v8 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v11 = (_QWORD *)(v9 + 32LL * v10);
  v12 = *v11;
  if ( a2 != *v11 )
  {
    v19 = 1;
    v20 = 0;
    while ( v12 != -8 )
    {
      if ( !v20 && v12 == -16 )
        v20 = v11;
      v10 = (v8 - 1) & (v19 + v10);
      v11 = (_QWORD *)(v9 + 32LL * v10);
      v12 = *v11;
      if ( *v11 == a2 )
        goto LABEL_3;
      ++v19;
    }
    v21 = *(_DWORD *)(a1 + 16);
    if ( !v20 )
      v20 = v11;
    ++*(_QWORD *)a1;
    v22 = v21 + 1;
    if ( 4 * (v21 + 1) < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(a1 + 20) - v22 > v8 >> 3 )
        goto LABEL_12;
      sub_1384670(a1, v8);
      v31 = *(_DWORD *)(a1 + 24);
      if ( v31 )
      {
        v32 = v31 - 1;
        v33 = *(_QWORD *)(a1 + 8);
        v34 = 1;
        v30 = 0;
        v35 = (v31 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
        v22 = *(_DWORD *)(a1 + 16) + 1;
        v20 = (_QWORD *)(v33 + 32LL * v35);
        v36 = *v20;
        if ( a2 == *v20 )
          goto LABEL_12;
        while ( v36 != -8 )
        {
          if ( !v30 && v36 == -16 )
            v30 = v20;
          v35 = v32 & (v34 + v35);
          v20 = (_QWORD *)(v33 + 32LL * v35);
          v36 = *v20;
          if ( *v20 == a2 )
            goto LABEL_12;
          ++v34;
        }
LABEL_23:
        if ( v30 )
          v20 = v30;
LABEL_12:
        *(_DWORD *)(a1 + 16) = v22;
        if ( *v20 != -8 )
          --*(_DWORD *)(a1 + 20);
        v23 = a3 + 1;
        *v20 = a2;
        v16 = 0;
        v15 = a3;
        v20[1] = 0;
        v20[2] = 0;
        v20[3] = 0;
        if ( a3 == -1 )
          goto LABEL_17;
LABEL_15:
        v40 = v20;
        sub_1383AB0(v20 + 1, v23 - v16);
        v17 = 1;
        v14 = v40[1];
        goto LABEL_5;
      }
      goto LABEL_55;
    }
LABEL_19:
    sub_1384670(a1, 2 * v8);
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 8);
      v27 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v22 = *(_DWORD *)(a1 + 16) + 1;
      v20 = (_QWORD *)(v26 + 32LL * v27);
      v28 = *v20;
      if ( *v20 == a2 )
        goto LABEL_12;
      v29 = 1;
      v30 = 0;
      while ( v28 != -8 )
      {
        if ( !v30 && v28 == -16 )
          v30 = v20;
        v27 = v25 & (v29 + v27);
        v20 = (_QWORD *)(v26 + 32LL * v27);
        v28 = *v20;
        if ( *v20 == a2 )
          goto LABEL_12;
        ++v29;
      }
      goto LABEL_23;
    }
LABEL_55:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_3:
  v13 = (_QWORD *)v11[2];
  v14 = v11[1];
  v15 = a3;
  v16 = 0x6DB6DB6DB6DB6DB7LL * (((__int64)v13 - v14) >> 3);
  if ( v16 > a3 )
  {
    v17 = 0;
    goto LABEL_5;
  }
  v23 = a3 + 1;
  if ( v16 < v23 )
  {
    v20 = v11;
    goto LABEL_15;
  }
  if ( v16 <= v23 )
  {
    v20 = v11;
LABEL_17:
    v14 = v20[1];
    v17 = 1;
    goto LABEL_5;
  }
  v37 = (_QWORD *)(v14 + 56 * v23);
  v38 = v37;
  if ( v13 != v37 )
  {
    do
    {
      v39 = v38[3];
      if ( v39 )
      {
        v41 = v13;
        j_j___libc_free_0(v39, v38[5] - v39);
        v13 = v41;
      }
      if ( *v38 )
      {
        v42 = v13;
        j_j___libc_free_0(*v38, v38[2] - *v38);
        v13 = v42;
      }
      v38 += 7;
    }
    while ( v13 != v38 );
    v11[2] = v37;
    v14 = v11[1];
  }
  v17 = 1;
LABEL_5:
  *(_QWORD *)(v14 + 56 * v15 + 48) |= a4;
  return v17;
}
