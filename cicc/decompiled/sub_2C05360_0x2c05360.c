// Function: sub_2C05360
// Address: 0x2c05360
//
_QWORD *__fastcall sub_2C05360(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  _QWORD *v6; // rax
  _QWORD *v7; // r12
  int v8; // eax
  unsigned int v9; // esi
  unsigned int v10; // r15d
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r14
  __int64 v16; // rdx
  unsigned __int64 *v17; // rax
  unsigned __int64 v18; // r14
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  int v24; // ecx
  bool v25; // cf
  __int64 v26; // r15
  unsigned __int64 v27; // rdx
  _QWORD *v28; // rax
  _QWORD *v29; // r15
  __int64 v30; // rcx
  int v31; // eax
  int v32; // edi
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // rbx
  unsigned __int64 v36; // r12
  unsigned __int64 v37; // rdi
  int v38; // eax
  int v39; // esi
  __int64 v40; // rdx
  unsigned int v41; // eax
  int v42; // r10d
  int v43; // eax
  int v44; // eax
  int v45; // r10d
  unsigned int v46; // edx
  __int64 v47; // rsi
  __int64 v48; // [rsp+8h] [rbp-48h]
  __int64 v49; // [rsp+10h] [rbp-40h]
  _QWORD *v50; // [rsp+10h] [rbp-40h]
  int v51; // [rsp+1Ch] [rbp-34h]
  unsigned int v52; // [rsp+1Ch] [rbp-34h]

  v5 = a1;
  v6 = (_QWORD *)sub_22077B0(0x50u);
  v7 = v6;
  if ( v6 )
  {
    *v6 = a2;
    v6[1] = a3;
    v8 = 0;
    if ( a3 )
      v8 = *(_DWORD *)(a3 + 16) + 1;
    *((_DWORD *)v7 + 4) = v8;
    v7[3] = v7 + 5;
    v7[4] = 0x400000000LL;
    v7[9] = -1;
  }
  v9 = *(_DWORD *)(a1 + 112);
  v10 = *(_DWORD *)(a1 + 32);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 88);
    goto LABEL_47;
  }
  v11 = *(_QWORD *)(a1 + 96);
  v12 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = v11 + 16 * v12;
  v14 = *(_QWORD *)v13;
  if ( *(_QWORD *)v13 == a2 )
  {
LABEL_7:
    v15 = *(unsigned int *)(v13 + 8);
    goto LABEL_8;
  }
  v51 = 1;
  v30 = 0;
  v49 = v11;
  while ( v14 != -4096 )
  {
    if ( v14 == -8192 && !v30 )
      v30 = v13;
    v12 = (v9 - 1) & (v51 + (_DWORD)v12);
    v11 = (unsigned int)(v51 + 1);
    v13 = v49 + 16LL * (unsigned int)v12;
    v14 = *(_QWORD *)v13;
    if ( *(_QWORD *)v13 == a2 )
      goto LABEL_7;
    ++v51;
  }
  if ( !v30 )
    v30 = v13;
  v31 = *(_DWORD *)(v5 + 104);
  ++*(_QWORD *)(v5 + 88);
  v32 = v31 + 1;
  if ( 4 * (v31 + 1) >= 3 * v9 )
  {
LABEL_47:
    sub_2C05180(v5 + 88, 2 * v9);
    v38 = *(_DWORD *)(v5 + 112);
    if ( v38 )
    {
      v39 = v38 - 1;
      v40 = *(_QWORD *)(v5 + 96);
      v41 = (v38 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v32 = *(_DWORD *)(v5 + 104) + 1;
      v30 = v40 + 16LL * v41;
      v12 = *(_QWORD *)v30;
      if ( *(_QWORD *)v30 == a2 )
        goto LABEL_35;
      v42 = 1;
      v11 = 0;
      while ( v12 != -4096 )
      {
        if ( !v11 && v12 == -8192 )
          v11 = v30;
        v41 = v39 & (v42 + v41);
        v30 = v40 + 16LL * v41;
        v12 = *(_QWORD *)v30;
        if ( *(_QWORD *)v30 == a2 )
          goto LABEL_35;
        ++v42;
      }
LABEL_51:
      if ( v11 )
        v30 = v11;
      goto LABEL_35;
    }
LABEL_72:
    ++*(_DWORD *)(v5 + 104);
    BUG();
  }
  v12 = v9 >> 3;
  if ( v9 - *(_DWORD *)(v5 + 108) - v32 <= (unsigned int)v12 )
  {
    v52 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    sub_2C05180(v5 + 88, v9);
    v43 = *(_DWORD *)(v5 + 112);
    if ( v43 )
    {
      v44 = v43 - 1;
      v12 = *(_QWORD *)(v5 + 96);
      v45 = 1;
      v11 = 0;
      v46 = v44 & v52;
      v32 = *(_DWORD *)(v5 + 104) + 1;
      v30 = v12 + 16LL * (v44 & v52);
      v47 = *(_QWORD *)v30;
      if ( *(_QWORD *)v30 == a2 )
        goto LABEL_35;
      while ( v47 != -4096 )
      {
        if ( v47 == -8192 && !v11 )
          v11 = v30;
        v46 = v44 & (v45 + v46);
        v30 = v12 + 16LL * v46;
        v47 = *(_QWORD *)v30;
        if ( *(_QWORD *)v30 == a2 )
          goto LABEL_35;
        ++v45;
      }
      goto LABEL_51;
    }
    goto LABEL_72;
  }
LABEL_35:
  *(_DWORD *)(v5 + 104) = v32;
  if ( *(_QWORD *)v30 != -4096 )
    --*(_DWORD *)(v5 + 108);
  *(_QWORD *)v30 = a2;
  v15 = v10;
  *(_DWORD *)(v30 + 8) = v10;
  v10 = *(_DWORD *)(v5 + 32);
LABEL_8:
  if ( v10 > (unsigned int)v15 || (v22 = (unsigned int)(v15 + 1), v23 = v10, v24 = v15 + 1, v25 = v22 < v10, v22 == v10) )
  {
    v16 = *(_QWORD *)(v5 + 24);
  }
  else
  {
    v26 = 8 * v22;
    if ( v25 )
    {
      v16 = *(_QWORD *)(v5 + 24);
      v33 = v16 + 8 * v23;
      v34 = v16 + v26;
      if ( v33 != v34 )
      {
        v50 = v7;
        v48 = v5;
        v35 = v33;
        do
        {
          v36 = *(_QWORD *)(v35 - 8);
          v35 -= 8;
          if ( v36 )
          {
            v37 = *(_QWORD *)(v36 + 24);
            if ( v37 != v36 + 40 )
              _libc_free(v37);
            j_j___libc_free_0(v36);
          }
        }
        while ( v34 != v35 );
        v5 = v48;
        v24 = v15 + 1;
        v7 = v50;
        v16 = *(_QWORD *)(v48 + 24);
      }
    }
    else
    {
      v27 = *(unsigned int *)(v5 + 36);
      if ( v22 > v27 )
      {
        sub_2C043A0(v5 + 24, v22, v27, v22, v12, v11);
        v23 = *(unsigned int *)(v5 + 32);
        v24 = v15 + 1;
      }
      v16 = *(_QWORD *)(v5 + 24);
      v28 = (_QWORD *)(v16 + 8 * v23);
      v29 = (_QWORD *)(v16 + v26);
      if ( v28 != v29 )
      {
        do
        {
          if ( v28 )
            *v28 = 0;
          ++v28;
        }
        while ( v29 != v28 );
        v16 = *(_QWORD *)(v5 + 24);
      }
    }
    *(_DWORD *)(v5 + 32) = v24;
  }
  v17 = (unsigned __int64 *)(v16 + 8 * v15);
  v18 = *v17;
  *v17 = (unsigned __int64)v7;
  if ( v18 )
  {
    v19 = *(_QWORD *)(v18 + 24);
    if ( v19 != v18 + 40 )
      _libc_free(v19);
    j_j___libc_free_0(v18);
  }
  if ( a3 )
  {
    v20 = *(unsigned int *)(a3 + 32);
    if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 36) )
    {
      sub_C8D5F0(a3 + 24, (const void *)(a3 + 40), v20 + 1, 8u, v12, v11);
      v20 = *(unsigned int *)(a3 + 32);
    }
    *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v20) = v7;
    ++*(_DWORD *)(a3 + 32);
  }
  return v7;
}
