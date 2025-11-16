// Function: sub_22BF7F0
// Address: 0x22bf7f0
//
void __fastcall sub_22BF7F0(__int64 a1, __int64 a2, __int64 a3)
{
  char *v4; // rax
  int v5; // eax
  int v6; // edx
  __int64 v7; // rsi
  unsigned int v8; // eax
  __int64 v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 *v17; // r15
  signed __int64 v18; // r13
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 *v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 *v26; // rcx
  __int64 *v27; // rbx
  __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // rdi
  unsigned int v31; // esi
  __int64 v32; // rdx
  __int64 v33; // r9
  __int64 v34; // r15
  __int64 *v35; // r13
  __int64 *v36; // r12
  char v37; // bl
  __int64 v38; // rdi
  int v39; // r8d
  unsigned int v40; // edx
  _QWORD *v41; // rsi
  __int64 v42; // r9
  __int64 v43; // rax
  unsigned int v44; // eax
  __int64 v45; // rax
  int v46; // edx
  unsigned __int64 v47; // rax
  __int64 v48; // r13
  __int64 v49; // rax
  int v50; // r15d
  __int64 v51; // r12
  __int64 i; // r15
  int v53; // esi
  int v54; // r10d
  size_t v55; // r12
  unsigned __int64 v56; // rcx
  bool v57; // cf
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rbx
  __int64 *v60; // r12
  unsigned int v61; // ebx
  int v62; // edx
  int v63; // r10d
  int v64; // edi
  char *v65; // [rsp+8h] [rbp-E8h]
  __int64 *v66; // [rsp+18h] [rbp-D8h]
  char *v67; // [rsp+18h] [rbp-D8h]
  _QWORD *dest; // [rsp+20h] [rbp-D0h]
  char *desta; // [rsp+20h] [rbp-D0h]
  __int64 v70; // [rsp+28h] [rbp-C8h]
  char *src; // [rsp+40h] [rbp-B0h]
  char *v74; // [rsp+48h] [rbp-A8h]
  __int64 *v75; // [rsp+48h] [rbp-A8h]
  _QWORD v76[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v77; // [rsp+60h] [rbp-90h]
  _QWORD v78[2]; // [rsp+70h] [rbp-80h] BYREF
  __int64 v79; // [rsp+80h] [rbp-70h]
  __int64 *v80; // [rsp+90h] [rbp-60h] BYREF
  __int64 v81; // [rsp+98h] [rbp-58h] BYREF
  _QWORD v82[2]; // [rsp+A0h] [rbp-50h] BYREF
  char v83; // [rsp+B0h] [rbp-40h]

  v4 = (char *)sub_22077B0(8u);
  src = v4;
  if ( v4 )
    *(_QWORD *)v4 = a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
    goto LABEL_8;
  v6 = v5 - 1;
  v7 = *(_QWORD *)(a1 + 8);
  v83 = 0;
  v81 = 2;
  v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v82[0] = 0;
  v82[1] = -4096;
  v9 = v7 + 48LL * v8;
  v10 = *(_QWORD *)(v9 + 24);
  if ( v10 != a2 )
  {
    v64 = 1;
    while ( v10 != -4096 )
    {
      v8 = v6 & (v64 + v8);
      v9 = v7 + 48LL * v8;
      v10 = *(_QWORD *)(v9 + 24);
      if ( v10 == a2 )
        goto LABEL_5;
      ++v64;
    }
    v80 = (__int64 *)&unk_49DB368;
    sub_D68D70(&v81);
LABEL_8:
    if ( src )
      goto LABEL_9;
    return;
  }
LABEL_5:
  v80 = (__int64 *)&unk_49DB368;
  sub_D68D70(&v81);
  if ( v9 == *(_QWORD *)(a1 + 8) + 48LL * *(unsigned int *)(a1 + 24) )
    goto LABEL_8;
  v13 = *(_QWORD *)(v9 + 40);
  if ( !v13 || !(*(_DWORD *)(v13 + 280) >> 1) )
    goto LABEL_8;
  if ( (*(_BYTE *)(v13 + 280) & 1) != 0 )
  {
    v14 = v13 + 288;
    v15 = v13 + 384;
  }
  else
  {
    v14 = *(_QWORD *)(v13 + 288);
    v15 = v14 + 24LL * *(unsigned int *)(v13 + 296);
  }
  if ( v14 == v15 )
  {
LABEL_16:
    HIDWORD(v81) = 4;
    v80 = v82;
  }
  else
  {
    while ( 1 )
    {
      v16 = *(_QWORD *)(v14 + 16);
      if ( v16 != -4096 && v16 != -8192 )
        break;
      v14 += 24;
      if ( v14 == v15 )
        goto LABEL_16;
    }
    v80 = v82;
    v81 = 0x400000000LL;
    if ( v15 != v14 )
    {
      v20 = v14;
      v18 = 0;
      while ( 1 )
      {
        v21 = v20 + 24;
        if ( v20 + 24 == v15 )
          break;
        while ( 1 )
        {
          v22 = *(_QWORD *)(v21 + 16);
          v20 = v21;
          if ( v22 != -4096 && v22 != -8192 )
            break;
          v21 += 24;
          if ( v15 == v21 )
            goto LABEL_25;
        }
        ++v18;
        if ( v15 == v21 )
          goto LABEL_26;
      }
LABEL_25:
      ++v18;
LABEL_26:
      v23 = v82;
      if ( v18 > 4 )
      {
        sub_C8D5F0((__int64)&v80, v82, v18, 8u, v11, v12);
        v23 = &v80[(unsigned int)v81];
      }
      v24 = *(_QWORD *)(v14 + 16);
      do
      {
        if ( v23 )
          *v23 = v24;
        v25 = v14 + 24;
        if ( v14 + 24 == v15 )
          break;
        while ( 1 )
        {
          v24 = *(_QWORD *)(v25 + 16);
          v14 = v25;
          if ( v24 != -8192 && v24 != -4096 )
            break;
          v25 += 24;
          if ( v15 == v25 )
            goto LABEL_35;
        }
        ++v23;
      }
      while ( v15 != v25 );
LABEL_35:
      v19 = v81;
      v17 = v80;
      goto LABEL_36;
    }
  }
  v17 = v82;
  LODWORD(v18) = 0;
  v19 = 0;
LABEL_36:
  v26 = v17;
  LODWORD(v81) = v18 + v19;
  v27 = (__int64 *)(src + 8);
  v65 = src + 8;
  do
  {
    while ( 1 )
    {
      v28 = *(v27 - 1);
      v74 = (char *)(v27 - 1);
      if ( v28 == a3 )
        goto LABEL_37;
      v29 = *(unsigned int *)(a1 + 24);
      if ( !(_DWORD)v29 )
        goto LABEL_37;
      v30 = *(_QWORD *)(a1 + 8);
      v31 = (v29 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v32 = v30 + 48LL * v31;
      v33 = *(_QWORD *)(v32 + 24);
      if ( v28 == v33 )
        break;
      v62 = 1;
      while ( v33 != -4096 )
      {
        v63 = v62 + 1;
        v31 = (v29 - 1) & (v31 + v62);
        v32 = v30 + 48LL * v31;
        v33 = *(_QWORD *)(v32 + 24);
        if ( v28 == v33 )
          goto LABEL_41;
        v62 = v63;
      }
LABEL_37:
      v27 = (__int64 *)v74;
      if ( src == v74 )
        goto LABEL_79;
    }
LABEL_41:
    if ( v32 == v30 + 48 * v29 )
      goto LABEL_37;
    v34 = *(_QWORD *)(v32 + 40);
    if ( !(*(_DWORD *)(v34 + 280) >> 1) || &v26[(unsigned int)v81] == v26 )
      goto LABEL_37;
    dest = (_QWORD *)*(v27 - 1);
    v35 = &v26[(unsigned int)v81];
    v36 = v26;
    v66 = v27;
    v37 = 0;
    do
    {
      while ( 1 )
      {
        v45 = *v36;
        v76[0] = 0;
        v76[1] = 0;
        v77 = v45;
        if ( v45 != -4096 && v45 != 0 && v45 != -8192 )
        {
          sub_BD73F0((__int64)v76);
          v45 = v77;
        }
        if ( (*(_BYTE *)(v34 + 280) & 1) != 0 )
        {
          v38 = v34 + 288;
          v39 = 3;
        }
        else
        {
          v46 = *(_DWORD *)(v34 + 296);
          v38 = *(_QWORD *)(v34 + 288);
          v39 = v46 - 1;
          if ( !v46 )
            goto LABEL_65;
        }
        v40 = v39 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
        v41 = (_QWORD *)(v38 + 24LL * v40);
        v42 = v41[2];
        if ( v42 == v45 )
        {
LABEL_47:
          v79 = -8192;
          v43 = v41[2];
          v78[0] = 0;
          v78[1] = 0;
          if ( v43 != -8192 )
          {
            if ( v43 != -4096 && v43 )
              sub_BD60C0(v41);
            v41[2] = -8192;
            if ( v79 != -4096 && v79 != 0 && v79 != -8192 )
              sub_BD60C0(v78);
          }
          v44 = *(_DWORD *)(v34 + 280);
          ++*(_DWORD *)(v34 + 284);
          v37 = 1;
          *(_DWORD *)(v34 + 280) = (2 * (v44 >> 1) - 2) | v44 & 1;
          if ( v77 )
          {
            if ( v77 != -8192 && v77 != -4096 )
              sub_BD60C0(v76);
            v37 = 1;
          }
          goto LABEL_59;
        }
        v53 = 1;
        while ( v42 != -4096 )
        {
          v54 = v53 + 1;
          v40 = v39 & (v53 + v40);
          v41 = (_QWORD *)(v38 + 24LL * v40);
          v42 = v41[2];
          if ( v42 == v45 )
            goto LABEL_47;
          v53 = v54;
        }
LABEL_65:
        if ( v45 && v45 != -4096 && v45 != -8192 )
          break;
LABEL_59:
        if ( v35 == ++v36 )
          goto LABEL_69;
      }
      ++v36;
      sub_BD60C0(v76);
    }
    while ( v35 != v36 );
LABEL_69:
    v26 = v80;
    if ( !v37 )
      goto LABEL_37;
    v47 = dest[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v47 == dest + 6 )
      goto LABEL_113;
    if ( !v47 )
      BUG();
    v48 = v47 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v47 - 24) - 30 > 0xA || (LODWORD(v49) = sub_B46E30(v48), (v50 = v49) == 0) )
    {
LABEL_113:
      v27 = (__int64 *)v74;
      goto LABEL_78;
    }
    v70 = (int)v49;
    if ( (int)v49 <= (unsigned __int64)((v65 - v74) >> 3) )
    {
      v51 = (unsigned int)v49;
      for ( i = 0; i != v51; ++i )
        v66[i - 1] = sub_B46EC0(v48, i);
      v27 = (__int64 *)&v74[8 * v70];
      goto LABEL_78;
    }
    v55 = v74 - src;
    v56 = (v74 - src) >> 3;
    if ( (int)v49 > 0xFFFFFFFFFFFFFFFLL - v56 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    v49 = (int)v49;
    if ( (int)v49 < v56 )
      v49 = (v74 - src) >> 3;
    v57 = __CFADD__(v56, v49);
    v58 = v56 + v49;
    if ( v57 )
    {
      v59 = 0x7FFFFFFFFFFFFFF8LL;
      goto LABEL_99;
    }
    if ( v58 )
    {
      if ( v58 > 0xFFFFFFFFFFFFFFFLL )
        v58 = 0xFFFFFFFFFFFFFFFLL;
      v59 = 8 * v58;
LABEL_99:
      desta = (char *)sub_22077B0(v59);
      v67 = &desta[v59];
    }
    else
    {
      v67 = 0;
      desta = 0;
    }
    if ( v74 != src )
      memmove(desta, src, v55);
    v60 = (__int64 *)&desta[v55];
    v61 = 0;
    v75 = v60;
    do
    {
      if ( v60 )
        *v60 = sub_B46EC0(v48, v61);
      ++v61;
      ++v60;
    }
    while ( v50 != v61 );
    v27 = &v75[v70];
    if ( src )
      j_j___libc_free_0((unsigned __int64)src);
    v65 = v67;
    src = desta;
LABEL_78:
    v26 = v80;
  }
  while ( src != (char *)v27 );
LABEL_79:
  if ( v26 != v82 )
    _libc_free((unsigned __int64)v26);
LABEL_9:
  j_j___libc_free_0((unsigned __int64)src);
}
