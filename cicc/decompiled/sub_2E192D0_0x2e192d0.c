// Function: sub_2E192D0
// Address: 0x2e192d0
//
unsigned __int64 __fastcall sub_2E192D0(__int64 a1, unsigned __int64 a2, char a3)
{
  __int64 v5; // rsi
  __int64 v6; // r8
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  unsigned int v10; // edi
  __int64 *v11; // rcx
  __int64 v12; // r11
  __int64 v13; // r14
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // r8
  unsigned __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // r8
  unsigned int v22; // r9d
  __int64 *v23; // rcx
  __int64 v24; // r11
  __int64 v25; // rax
  int v26; // eax
  int v27; // r13d
  unsigned int v28; // r13d
  int v29; // r9d
  unsigned __int64 v30; // r15
  unsigned __int64 v31; // rsi
  __int64 v32; // rax
  unsigned int v33; // esi
  __int64 v34; // rdi
  __int64 v35; // rcx
  unsigned int v36; // edx
  unsigned __int64 *v37; // rax
  unsigned __int64 v38; // r11
  int v40; // ecx
  int v41; // r13d
  int v42; // ecx
  int v43; // r13d
  __int64 v44; // rcx
  int v45; // r8d
  int v46; // r8d
  __int64 v47; // r10
  int v48; // edx
  unsigned int v49; // eax
  unsigned __int64 *v50; // r9
  unsigned __int64 v51; // rdi
  int v52; // esi
  unsigned __int64 *v53; // rcx
  __int64 v54; // rax
  int v55; // r14d
  int v56; // eax
  int v57; // edi
  int v58; // edi
  __int64 v59; // r8
  unsigned __int64 *v60; // rsi
  __int64 v61; // r13
  int v62; // eax
  unsigned __int64 v63; // rcx
  unsigned __int64 v64; // [rsp+0h] [rbp-40h]
  int v65; // [rsp+Ch] [rbp-34h]

  v5 = *(_QWORD *)(a2 + 24);
  if ( !a3 )
  {
    v17 = *(_QWORD *)(v5 + 56);
    v18 = a2;
    while ( v17 != v18 )
    {
      v18 = *(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v18 )
LABEL_86:
        BUG();
      v19 = *(_QWORD *)v18;
      if ( (*(_QWORD *)v18 & 4) == 0 && (*(_BYTE *)(v18 + 44) & 4) != 0 )
      {
        while ( 1 )
        {
          v18 = v19 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)((v19 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
            break;
          v19 = *(_QWORD *)v18;
        }
      }
      v20 = *(unsigned int *)(a1 + 144);
      v21 = *(_QWORD *)(a1 + 128);
      if ( (_DWORD)v20 )
      {
        v22 = (v20 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v23 = (__int64 *)(v21 + 16LL * v22);
        v24 = *v23;
        if ( v18 == *v23 )
        {
LABEL_24:
          if ( v23 != (__int64 *)(v21 + 16 * v20) )
          {
            v25 = v23[1];
            goto LABEL_27;
          }
        }
        else
        {
          v42 = 1;
          while ( v24 != -4096 )
          {
            v43 = v42 + 1;
            v44 = ((_DWORD)v20 - 1) & (v22 + v42);
            v22 = v44;
            v23 = (__int64 *)(v21 + 16 * v44);
            v24 = *v23;
            if ( v18 == *v23 )
              goto LABEL_24;
            v42 = v43;
          }
        }
      }
    }
    v25 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 16LL * *(unsigned int *)(v5 + 24));
LABEL_27:
    v16 = v25 & 0xFFFFFFFFFFFFFFF8LL;
    v14 = *(_QWORD *)(v16 + 8);
    v15 = v14;
    goto LABEL_28;
  }
  v6 = v5 + 48;
  v7 = a2;
  while ( 1 )
  {
    if ( (*(_BYTE *)v7 & 4) != 0 )
    {
      v7 = *(_QWORD *)(v7 + 8);
      if ( v6 == v7 )
        goto LABEL_10;
    }
    else
    {
      while ( (*(_BYTE *)(v7 + 44) & 8) != 0 )
        v7 = *(_QWORD *)(v7 + 8);
      v7 = *(_QWORD *)(v7 + 8);
      if ( v6 == v7 )
      {
LABEL_10:
        v13 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 16LL * *(unsigned int *)(v5 + 24) + 8);
        goto LABEL_11;
      }
    }
    v8 = *(unsigned int *)(a1 + 144);
    v9 = *(_QWORD *)(a1 + 128);
    if ( (_DWORD)v8 )
    {
      v10 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( *v11 != v7 )
      {
        v40 = 1;
        while ( v12 != -4096 )
        {
          v41 = v40 + 1;
          v10 = (v8 - 1) & (v40 + v10);
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( *v11 == v7 )
            goto LABEL_6;
          v40 = v41;
        }
        goto LABEL_7;
      }
LABEL_6:
      if ( v11 != (__int64 *)(v9 + 16 * v8) )
        break;
    }
LABEL_7:
    if ( !v7 )
      goto LABEL_86;
  }
  v13 = v11[1];
LABEL_11:
  v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
  v15 = v14;
  v16 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_28:
  v26 = *(_DWORD *)(v16 + 24);
  v27 = *(_DWORD *)(v14 + 24);
  *(_QWORD *)(a1 + 80) += 32LL;
  v28 = ((unsigned int)(v27 - v26) >> 1) & 0xFFFFFFFC;
  v29 = v28 + v26;
  v30 = (*(_QWORD *)a1 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(a1 + 8) >= v30 + 32 && *(_QWORD *)a1 )
  {
    *(_QWORD *)a1 = v30 + 32;
    v31 = v30;
  }
  else
  {
    v64 = v15;
    v65 = v28 + v26;
    v54 = sub_9D1E70(a1, 32, 32, 3);
    v15 = v64;
    v29 = v65;
    v31 = v54;
    v30 = v54 & 0xFFFFFFFFFFFFFFF9LL;
  }
  *(_QWORD *)v31 = 0;
  *(_QWORD *)(v31 + 8) = 0;
  *(_QWORD *)(v31 + 16) = a2;
  *(_DWORD *)(v31 + 24) = v29;
  v32 = *(_QWORD *)v14;
  *(_QWORD *)(v31 + 8) = v15;
  v32 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v31 = v32;
  *(_QWORD *)(v32 + 8) = v31;
  *(_QWORD *)v14 = v31 | *(_QWORD *)v14 & 7LL;
  if ( v28 )
  {
    v33 = *(_DWORD *)(a1 + 144);
    v34 = a1 + 120;
    if ( v33 )
      goto LABEL_33;
LABEL_44:
    ++*(_QWORD *)(a1 + 120);
LABEL_45:
    sub_2E190F0(v34, 2 * v33);
    v45 = *(_DWORD *)(a1 + 144);
    if ( v45 )
    {
      v46 = v45 - 1;
      v47 = *(_QWORD *)(a1 + 128);
      v48 = *(_DWORD *)(a1 + 136) + 1;
      v49 = v46 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v50 = (unsigned __int64 *)(v47 + 16LL * v49);
      v51 = *v50;
      if ( a2 != *v50 )
      {
        v52 = 1;
        v53 = 0;
        while ( v51 != -4096 )
        {
          if ( v51 == -8192 && !v53 )
            v53 = v50;
          v49 = v46 & (v52 + v49);
          v50 = (unsigned __int64 *)(v47 + 16LL * v49);
          v51 = *v50;
          if ( a2 == *v50 )
            goto LABEL_61;
          ++v52;
        }
        if ( v53 )
          v50 = v53;
      }
      goto LABEL_61;
    }
LABEL_85:
    ++*(_DWORD *)(a1 + 136);
    BUG();
  }
  sub_2FAD5B0(a1);
  v33 = *(_DWORD *)(a1 + 144);
  v34 = a1 + 120;
  if ( !v33 )
    goto LABEL_44;
LABEL_33:
  v35 = *(_QWORD *)(a1 + 128);
  v36 = (v33 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v37 = (unsigned __int64 *)(v35 + 16LL * v36);
  v38 = *v37;
  if ( a2 != *v37 )
  {
    v55 = 1;
    v50 = 0;
    while ( v38 != -4096 )
    {
      if ( v38 == -8192 && !v50 )
        v50 = v37;
      v36 = (v33 - 1) & (v55 + v36);
      v37 = (unsigned __int64 *)(v35 + 16LL * v36);
      v38 = *v37;
      if ( a2 == *v37 )
        return v30;
      ++v55;
    }
    if ( !v50 )
      v50 = v37;
    v56 = *(_DWORD *)(a1 + 136);
    ++*(_QWORD *)(a1 + 120);
    v48 = v56 + 1;
    if ( 4 * (v56 + 1) >= 3 * v33 )
      goto LABEL_45;
    if ( v33 - *(_DWORD *)(a1 + 140) - v48 <= v33 >> 3 )
    {
      sub_2E190F0(v34, v33);
      v57 = *(_DWORD *)(a1 + 144);
      if ( v57 )
      {
        v58 = v57 - 1;
        v59 = *(_QWORD *)(a1 + 128);
        v60 = 0;
        LODWORD(v61) = v58 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v48 = *(_DWORD *)(a1 + 136) + 1;
        v62 = 1;
        v50 = (unsigned __int64 *)(v59 + 16LL * (unsigned int)v61);
        v63 = *v50;
        if ( a2 != *v50 )
        {
          while ( v63 != -4096 )
          {
            if ( v63 == -8192 && !v60 )
              v60 = v50;
            v61 = v58 & (unsigned int)(v61 + v62);
            v50 = (unsigned __int64 *)(v59 + 16 * v61);
            v63 = *v50;
            if ( a2 == *v50 )
              goto LABEL_61;
            ++v62;
          }
          if ( v60 )
            v50 = v60;
        }
        goto LABEL_61;
      }
      goto LABEL_85;
    }
LABEL_61:
    *(_DWORD *)(a1 + 136) = v48;
    if ( *v50 != -4096 )
      --*(_DWORD *)(a1 + 140);
    *v50 = a2;
    v50[1] = v30;
  }
  return v30;
}
