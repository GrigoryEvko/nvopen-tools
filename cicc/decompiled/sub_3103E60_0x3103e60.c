// Function: sub_3103E60
// Address: 0x3103e60
//
void __fastcall sub_3103E60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned int v7; // esi
  __int64 v8; // rcx
  unsigned int v9; // edi
  int v10; // r11d
  _QWORD *v11; // r9
  unsigned int v12; // r8d
  _QWORD *v13; // r12
  __int64 v14; // rax
  __int64 *v15; // r12
  int v16; // r11d
  _QWORD *v17; // r9
  unsigned int v18; // r8d
  _QWORD *v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rcx
  unsigned __int64 v24; // r13
  __int64 v25; // rdx
  unsigned __int64 v26; // rdi
  __int64 v27; // r8
  __int64 v28; // rax
  unsigned __int64 v29; // rbx
  unsigned __int64 v30; // r12
  __int64 v31; // rax
  unsigned __int64 v32; // rdi
  int v33; // eax
  int v34; // ecx
  int v35; // eax
  int v36; // esi
  __int64 v37; // rdi
  unsigned int v38; // ecx
  int v39; // edx
  __int64 v40; // rax
  int v41; // r10d
  _QWORD *v42; // r8
  int v43; // eax
  _QWORD *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  unsigned __int64 v49; // rbx
  int v50; // eax
  int v51; // edx
  __int64 v52; // rsi
  unsigned int v53; // eax
  __int64 v54; // rdi
  int v55; // r10d
  _QWORD *v56; // r8
  int v57; // eax
  int v58; // ecx
  __int64 v59; // rsi
  int v60; // r8d
  unsigned int v61; // r13d
  _QWORD *v62; // rdi
  __int64 v63; // rax
  int v64; // eax
  int v65; // eax
  __int64 v66; // rdi
  int v67; // r10d
  unsigned int v68; // edx
  __int64 v69; // rsi
  unsigned int v70; // [rsp+Ch] [rbp-34h]

  v3 = a1 + 8;
  v7 = *(_DWORD *)(a1 + 32);
  if ( v7 )
  {
    v8 = *(_QWORD *)(a1 + 16);
    v9 = v7 - 1;
    v10 = 1;
    v11 = 0;
    v12 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = (_QWORD *)(v8 + 16LL * v12);
    v14 = *v13;
    if ( *v13 == a2 )
    {
LABEL_3:
      v15 = v13 + 1;
      goto LABEL_4;
    }
    while ( v14 != -4096 )
    {
      if ( v14 == -8192 && !v11 )
        v11 = v13;
      v12 = v9 & (v10 + v12);
      v13 = (_QWORD *)(v8 + 16LL * v12);
      v14 = *v13;
      if ( *v13 == a2 )
        goto LABEL_3;
      ++v10;
    }
    v33 = *(_DWORD *)(a1 + 24);
    if ( !v11 )
      v11 = v13;
    ++*(_QWORD *)(a1 + 8);
    v34 = v33 + 1;
    if ( 4 * (v33 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 28) - v34 > v7 >> 3 )
        goto LABEL_35;
      v70 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
      sub_B2ACE0(v3, v7);
      v64 = *(_DWORD *)(a1 + 32);
      if ( !v64 )
        goto LABEL_106;
      v65 = v64 - 1;
      v66 = *(_QWORD *)(a1 + 16);
      v67 = 1;
      v56 = 0;
      v68 = v65 & v70;
      v34 = *(_DWORD *)(a1 + 24) + 1;
      v11 = (_QWORD *)(v66 + 16LL * (v65 & v70));
      v69 = *v11;
      if ( *v11 == a2 )
        goto LABEL_35;
      while ( v69 != -4096 )
      {
        if ( v69 == -8192 && !v56 )
          v56 = v11;
        v68 = v65 & (v67 + v68);
        v11 = (_QWORD *)(v66 + 16LL * v68);
        v69 = *v11;
        if ( *v11 == a2 )
          goto LABEL_35;
        ++v67;
      }
      goto LABEL_74;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 8);
  }
  sub_B2ACE0(v3, 2 * v7);
  v50 = *(_DWORD *)(a1 + 32);
  if ( !v50 )
    goto LABEL_106;
  v51 = v50 - 1;
  v52 = *(_QWORD *)(a1 + 16);
  v53 = (v50 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v34 = *(_DWORD *)(a1 + 24) + 1;
  v11 = (_QWORD *)(v52 + 16LL * v53);
  v54 = *v11;
  if ( *v11 == a2 )
    goto LABEL_35;
  v55 = 1;
  v56 = 0;
  while ( v54 != -4096 )
  {
    if ( !v56 && v54 == -8192 )
      v56 = v11;
    v53 = v51 & (v55 + v53);
    v11 = (_QWORD *)(v52 + 16LL * v53);
    v54 = *v11;
    if ( *v11 == a2 )
      goto LABEL_35;
    ++v55;
  }
LABEL_74:
  if ( v56 )
    v11 = v56;
LABEL_35:
  *(_DWORD *)(a1 + 24) = v34;
  if ( *v11 != -4096 )
    --*(_DWORD *)(a1 + 28);
  *v11 = a2;
  v15 = v11 + 1;
  v11[1] = 0;
  v7 = *(_DWORD *)(a1 + 32);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_39;
  }
  v8 = *(_QWORD *)(a1 + 16);
  v9 = v7 - 1;
LABEL_4:
  v16 = 1;
  v17 = 0;
  v18 = v9 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v19 = (_QWORD *)(v8 + 16LL * v18);
  v20 = *v19;
  if ( *v19 == a3 )
  {
LABEL_5:
    v21 = v19 + 1;
    goto LABEL_6;
  }
  while ( v20 != -4096 )
  {
    if ( !v17 && v20 == -8192 )
      v17 = v19;
    v18 = v9 & (v16 + v18);
    v19 = (_QWORD *)(v8 + 16LL * v18);
    v20 = *v19;
    if ( *v19 == a3 )
      goto LABEL_5;
    ++v16;
  }
  if ( !v17 )
    v17 = v19;
  v43 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  v39 = v43 + 1;
  if ( 4 * (v43 + 1) >= 3 * v7 )
  {
LABEL_39:
    sub_B2ACE0(v3, 2 * v7);
    v35 = *(_DWORD *)(a1 + 32);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a1 + 16);
      v38 = (v35 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v39 = *(_DWORD *)(a1 + 24) + 1;
      v17 = (_QWORD *)(v37 + 16LL * v38);
      v40 = *v17;
      if ( *v17 != a3 )
      {
        v41 = 1;
        v42 = 0;
        while ( v40 != -4096 )
        {
          if ( !v42 && v40 == -8192 )
            v42 = v17;
          v38 = v36 & (v41 + v38);
          v17 = (_QWORD *)(v37 + 16LL * v38);
          v40 = *v17;
          if ( *v17 == a3 )
            goto LABEL_56;
          ++v41;
        }
        if ( v42 )
          v17 = v42;
      }
      goto LABEL_56;
    }
    goto LABEL_106;
  }
  if ( v7 - (v39 + *(_DWORD *)(a1 + 28)) <= v7 >> 3 )
  {
    sub_B2ACE0(v3, v7);
    v57 = *(_DWORD *)(a1 + 32);
    if ( v57 )
    {
      v58 = v57 - 1;
      v59 = *(_QWORD *)(a1 + 16);
      v60 = 1;
      v61 = (v57 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v39 = *(_DWORD *)(a1 + 24) + 1;
      v62 = 0;
      v17 = (_QWORD *)(v59 + 16LL * v61);
      v63 = *v17;
      if ( *v17 != a3 )
      {
        while ( v63 != -4096 )
        {
          if ( !v62 && v63 == -8192 )
            v62 = v17;
          v61 = v58 & (v60 + v61);
          v17 = (_QWORD *)(v59 + 16LL * v61);
          v63 = *v17;
          if ( *v17 == a3 )
            goto LABEL_56;
          ++v60;
        }
        if ( v62 )
          v17 = v62;
      }
      goto LABEL_56;
    }
LABEL_106:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
LABEL_56:
  *(_DWORD *)(a1 + 24) = v39;
  if ( *v17 != -4096 )
    --*(_DWORD *)(a1 + 28);
  *v17 = a3;
  v21 = v17 + 1;
  v17[1] = 0;
LABEL_6:
  if ( v21 == v15 )
    return;
  v22 = *v15;
  v23 = *v15 >> 2;
  v24 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v24 && ((v25 = *v21 & 4, !(_DWORD)v25) || (v25 = *(unsigned int *)(v24 + 8), (_DWORD)v25)) )
  {
    if ( (v23 & 1) != 0 )
    {
      v26 = v22 & 0xFFFFFFFFFFFFFFF8LL;
      v27 = (*v21 >> 2) & 1;
      if ( (_DWORD)v27 )
      {
        sub_3101CF0(v26, *v21 & 0xFFFFFFFFFFFFFFF8LL, v25, v23, v27, (__int64)v17);
      }
      else
      {
        *(_DWORD *)(v26 + 8) = 0;
        v28 = *v21;
        v29 = *v15 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v28 & 4) != 0 )
          v30 = **(_QWORD **)(v28 & 0xFFFFFFFFFFFFFFF8LL);
        else
          v30 = v28 & 0xFFFFFFFFFFFFFFF8LL;
        v31 = *(unsigned int *)(v29 + 8);
        if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(v29 + 12) )
        {
          sub_C8D5F0(v29, (const void *)(v29 + 16), v31 + 1, 8u, v27, (__int64)v17);
          v31 = *(unsigned int *)(v29 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v29 + 8 * v31) = v30;
        ++*(_DWORD *)(v29 + 8);
      }
      return;
    }
    if ( ((*v21 >> 2) & 1) == 0 )
    {
LABEL_63:
      *v15 = v24;
      return;
    }
    if ( *(_DWORD *)(v24 + 8) == 1 )
    {
      v24 = **(_QWORD **)v24 & 0xFFFFFFFFFFFFFFFBLL;
      goto LABEL_63;
    }
    v44 = (_QWORD *)sub_22077B0(0x30u);
    v49 = (unsigned __int64)v44;
    if ( v44 )
    {
      *v44 = v44 + 2;
      v44[1] = 0x400000000LL;
      if ( *(_DWORD *)(v24 + 8) )
        sub_3101CF0((__int64)v44, v24, v45, v46, v47, v48);
    }
    *v15 = v49 | 4;
  }
  else if ( (v23 & 1) != 0 )
  {
    if ( v22 )
    {
      if ( (v23 & 1) != 0 )
      {
        v32 = v22 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v32 )
          *(_DWORD *)(v32 + 8) = 0;
      }
    }
  }
  else
  {
    *v15 = 0;
  }
}
