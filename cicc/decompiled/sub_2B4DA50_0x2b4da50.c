// Function: sub_2B4DA50
// Address: 0x2b4da50
//
void __fastcall sub_2B4DA50(unsigned int **a1, __int64 a2, _QWORD *a3, char a4)
{
  unsigned __int64 v5; // r8
  _QWORD *v6; // r9
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // r15
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rbx
  unsigned __int8 **v14; // r12
  unsigned __int8 **v15; // rax
  unsigned __int8 **v16; // rdx
  int v17; // ebx
  unsigned __int8 **v18; // r15
  __int64 v19; // rbx
  _QWORD *v20; // r15
  __int64 v22; // r13
  unsigned __int8 *v23; // rsi
  int v24; // eax
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int8 ***v28; // r13
  _QWORD *v29; // r14
  __int64 v30; // rdi
  __int64 v31; // rsi
  unsigned __int8 *v32; // rdx
  unsigned int v33; // r11d
  _QWORD *v34; // rax
  __int64 v35; // rsi
  _QWORD *v36; // r10
  unsigned __int8 **v37; // rbx
  __int64 v38; // r10
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r10
  unsigned __int8 **v42; // r12
  unsigned __int8 *v43; // r15
  unsigned __int8 **v44; // r9
  unsigned __int8 *v45; // r15
  unsigned __int8 *v46; // rdi
  unsigned __int8 *v47; // r15
  unsigned __int8 *v48; // rdi
  int v49; // ecx
  unsigned int v50; // eax
  unsigned __int8 *v51; // rsi
  int *v52; // r12
  __int64 v53; // rax
  __int64 v54; // rbx
  __int64 v55; // r15
  unsigned __int8 *v56; // rdi
  int v57; // ecx
  __int64 v58; // rax
  unsigned __int8 *v59; // rsi
  unsigned __int8 *v60; // rdi
  int v61; // ecx
  __int64 v62; // rax
  unsigned __int8 *v63; // rsi
  int v64; // r8d
  unsigned __int8 *v65; // r8
  int v66; // ecx
  __int64 v67; // rax
  unsigned __int8 *v68; // rsi
  int v69; // ecx
  int v70; // ecx
  int v71; // ecx
  int v72; // ecx
  unsigned __int8 **v73; // rdx
  unsigned __int8 *v74; // rcx
  char v75; // al
  char v76; // al
  int v77; // r8d
  int v78; // r8d
  int v79; // r9d
  int *v80; // rbx
  _QWORD *v81; // r15
  unsigned __int8 ***v82; // r14
  __int64 v83; // r13
  int v84; // [rsp+18h] [rbp-C8h]
  __int64 v85; // [rsp+18h] [rbp-C8h]
  __int64 v86; // [rsp+18h] [rbp-C8h]
  int v87; // [rsp+18h] [rbp-C8h]
  int v88; // [rsp+24h] [rbp-BCh]
  bool v89; // [rsp+28h] [rbp-B8h]
  unsigned __int8 **v90; // [rsp+28h] [rbp-B8h]
  unsigned __int8 *v91; // [rsp+28h] [rbp-B8h]
  unsigned __int8 **v93; // [rsp+38h] [rbp-A8h]
  unsigned __int8 **v94; // [rsp+38h] [rbp-A8h]
  int *v95; // [rsp+38h] [rbp-A8h]
  unsigned __int8 *v96; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v97; // [rsp+48h] [rbp-98h]
  __int64 v98; // [rsp+50h] [rbp-90h] BYREF
  __int64 v99; // [rsp+58h] [rbp-88h]
  __int64 v100; // [rsp+60h] [rbp-80h]
  unsigned int v101; // [rsp+68h] [rbp-78h]
  int *v102; // [rsp+70h] [rbp-70h] BYREF
  __int64 v103; // [rsp+78h] [rbp-68h]
  _BYTE v104[96]; // [rsp+80h] [rbp-60h] BYREF

  if ( a4 && (v89 = sub_2B08550(*(unsigned __int8 ***)a2, *(unsigned int *)(a2 + 8))) )
  {
    if ( v5 <= 2 )
      v89 = v6[v5 - 1] == *v6;
  }
  else
  {
    v89 = 0;
  }
  v7 = sub_ACADE0(*(__int64 ***)a1[1]);
  v10 = *(unsigned int *)(a2 + 8);
  v11 = v7;
  v12 = **a1;
  v13 = v12 - v10;
  if ( v12 > *(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v12, 8u, v8, v9);
    v10 = *(unsigned int *)(a2 + 8);
  }
  v14 = *(unsigned __int8 ***)a2;
  if ( v13 )
  {
    v15 = &v14[v10];
    v16 = &v15[v13];
    if ( v15 != v16 )
    {
      do
        *v15++ = (unsigned __int8 *)v11;
      while ( v16 != v15 );
      v14 = *(unsigned __int8 ***)a2;
      v10 = *(unsigned int *)(a2 + 8);
    }
  }
  v17 = v10 + v13;
  v98 = 0;
  *(_DWORD *)(a2 + 8) = v17;
  v18 = &v14[v17];
  v102 = (int *)v104;
  v103 = 0xC00000000LL;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  if ( v18 == v14 )
  {
    v30 = 0;
    v31 = 0;
    goto LABEL_27;
  }
  v19 = 0;
  v93 = v18;
  v20 = a3;
  v84 = 0;
  v88 = 0;
  do
  {
    v23 = *v14;
    v24 = **v14;
    if ( v24 != 12 )
    {
      if ( v24 == 13 )
        goto LABEL_15;
      v22 = (unsigned int)v19;
      if ( (unsigned __int8)sub_2B0D8B0(*v14) )
      {
        *(_DWORD *)(*v20 + 4 * v19) = v19;
        goto LABEL_15;
      }
      v96 = v23;
      ++v88;
      v85 = v9;
      v27 = sub_ACADE0(*(__int64 ***)a1[1]);
      v9 = v85;
      v10 = (unsigned int)v19;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v19) = v27;
      if ( v89 )
      {
        **(_QWORD **)a2 = v96;
        *(_DWORD *)(*v20 + v85) = 0;
LABEL_22:
        v84 = v10;
        goto LABEL_15;
      }
      if ( v101 )
      {
        v32 = v96;
        v8 = v101 - 1;
        v33 = v8 & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
        v34 = (_QWORD *)(v99 + 16LL * v33);
        v35 = *v34;
        if ( v96 == (unsigned __int8 *)*v34 )
        {
LABEL_32:
          v22 = *((unsigned int *)v34 + 2);
LABEL_33:
          *(_QWORD *)(*(_QWORD *)a2 + 8 * v22) = v32;
          *(_DWORD *)(*v20 + v9) = *((_DWORD *)v34 + 2);
          goto LABEL_22;
        }
        v87 = 1;
        v36 = 0;
        while ( v35 != -4096 )
        {
          if ( !v36 && v35 == -8192 )
            v36 = v34;
          v33 = v8 & (v87 + v33);
          v34 = (_QWORD *)(v99 + 16LL * v33);
          v35 = *v34;
          if ( v96 == (unsigned __int8 *)*v34 )
            goto LABEL_32;
          ++v87;
        }
        if ( !v36 )
          v36 = v34;
      }
      else
      {
        v36 = 0;
      }
      v86 = v9;
      v34 = sub_10E8350((__int64)&v98, &v96, v36);
      v32 = v96;
      v10 = (unsigned int)v19;
      *((_DWORD *)v34 + 2) = v19;
      v9 = v86;
      *v34 = v32;
      goto LABEL_33;
    }
    *(_DWORD *)(*v20 + 4 * v19) = v19;
    v25 = (unsigned int)v103;
    v10 = HIDWORD(v103);
    v26 = (unsigned int)v103 + 1LL;
    if ( v26 > HIDWORD(v103) )
    {
      sub_C8D5F0((__int64)&v102, v104, v26, 4u, v8, v9);
      v25 = (unsigned int)v103;
    }
    v102[v25] = v19;
    LODWORD(v103) = v103 + 1;
LABEL_15:
    ++v19;
    ++v14;
  }
  while ( v93 != v14 );
  v28 = (unsigned __int8 ***)a2;
  v29 = v20;
  if ( v88 == 1 )
  {
    if ( v89 )
    {
      sub_11B1960((__int64)v20, **a1, -1, v10, v8, v9);
      v73 = &(*v28)[v84];
      v74 = **v28;
      **v28 = *v73;
      *v73 = v74;
      if ( (_DWORD)v103 )
      {
        if ( !*v102 )
          **v28 = (unsigned __int8 *)sub_ACA8A0(*(__int64 ***)a1[1]);
      }
    }
    *(_DWORD *)(*v20 + 4LL * v84) = v84;
    v30 = v99;
    v31 = 16LL * v101;
    goto LABEL_27;
  }
  if ( !(_DWORD)v103 || !v89 )
    goto LABEL_26;
  v37 = *v28;
  v38 = 8LL * *((unsigned int *)v28 + 2);
  v39 = *(_QWORD *)a1[3];
  v94 = &(*v28)[(unsigned __int64)v38 / 8];
  v96 = (unsigned __int8 *)a1[2];
  v97 = v39;
  v40 = v38 >> 3;
  v41 = v38 >> 5;
  if ( !v41 )
    goto LABEL_123;
  v42 = &v37[4 * v41];
  while ( 2 )
  {
    v47 = *v37;
    if ( (unsigned int)**v37 - 12 > 1 )
    {
      if ( (v96[88] & 1) != 0 )
      {
        v48 = v96 + 96;
        v49 = 3;
        goto LABEL_49;
      }
      v69 = *((_DWORD *)v96 + 26);
      v48 = (unsigned __int8 *)*((_QWORD *)v96 + 12);
      if ( v69 )
      {
        v49 = v69 - 1;
LABEL_49:
        v50 = v49 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
        v51 = *(unsigned __int8 **)&v48[72 * v50];
        if ( v51 == v47 )
          goto LABEL_50;
        v77 = 1;
        while ( v51 != (unsigned __int8 *)-4096LL )
        {
          v50 = v49 & (v77 + v50);
          v51 = *(unsigned __int8 **)&v48[72 * v50];
          if ( v47 == v51 )
            goto LABEL_50;
          ++v77;
        }
      }
      if ( sub_98ED70(*v37, *((_QWORD *)v96 + 416), 0, 0, 0)
        || *(_QWORD *)(v97 + 184) && sub_2B0E280(*((_QWORD *)v47 + 2), 0, v97) )
      {
        goto LABEL_50;
      }
    }
    v43 = v37[1];
    v44 = v37 + 1;
    if ( (unsigned int)*v43 - 12 > 1 )
    {
      if ( (v96[88] & 1) != 0 )
      {
        v56 = v96 + 96;
        v57 = 3;
      }
      else
      {
        v70 = *((_DWORD *)v96 + 26);
        v56 = (unsigned __int8 *)*((_QWORD *)v96 + 12);
        if ( !v70 )
          goto LABEL_88;
        v57 = v70 - 1;
      }
      LODWORD(v58) = v57 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v59 = *(unsigned __int8 **)&v56[72 * (unsigned int)v58];
      if ( v43 == v59 )
        goto LABEL_60;
      v78 = 1;
      while ( v59 != (unsigned __int8 *)-4096LL )
      {
        v58 = v57 & (unsigned int)(v58 + v78);
        v59 = *(unsigned __int8 **)&v56[72 * v58];
        if ( v43 == v59 )
          goto LABEL_60;
        ++v78;
      }
LABEL_88:
      v90 = v37 + 1;
      v75 = sub_98ED70(v37[1], *((_QWORD *)v96 + 416), 0, 0, 0);
      v44 = v37 + 1;
      if ( v75 )
        goto LABEL_60;
      if ( *(_QWORD *)(v97 + 184) && sub_2B0E280(*((_QWORD *)v43 + 2), 0, v97) )
        goto LABEL_91;
    }
    v45 = v37[2];
    v44 = v37 + 2;
    if ( (unsigned int)*v45 - 12 <= 1 )
      goto LABEL_44;
    if ( (v96[88] & 1) != 0 )
    {
      v60 = v96 + 96;
      v61 = 3;
      goto LABEL_63;
    }
    v71 = *((_DWORD *)v96 + 26);
    v60 = (unsigned __int8 *)*((_QWORD *)v96 + 12);
    if ( v71 )
    {
      v61 = v71 - 1;
LABEL_63:
      LODWORD(v62) = v61 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
      v63 = *(unsigned __int8 **)&v60[72 * (unsigned int)v62];
      if ( v63 == v45 )
        goto LABEL_60;
      v64 = 1;
      while ( v63 != (unsigned __int8 *)-4096LL )
      {
        v62 = v61 & (unsigned int)(v62 + v64);
        v63 = *(unsigned __int8 **)&v60[72 * v62];
        if ( v45 == v63 )
          goto LABEL_60;
        ++v64;
      }
    }
    v90 = v37 + 2;
    v76 = sub_98ED70(v37[2], *((_QWORD *)v96 + 416), 0, 0, 0);
    v44 = v37 + 2;
    if ( v76 )
    {
LABEL_60:
      v37 = v44;
      goto LABEL_50;
    }
    if ( *(_QWORD *)(v97 + 184) && sub_2B0E280(*((_QWORD *)v45 + 2), 0, v97) )
    {
LABEL_91:
      v37 = v90;
      goto LABEL_50;
    }
LABEL_44:
    v46 = v37[3];
    if ( (unsigned int)*v46 - 12 > 1 )
    {
      if ( (v96[88] & 1) != 0 )
      {
        v65 = v96 + 96;
        v66 = 3;
      }
      else
      {
        v72 = *((_DWORD *)v96 + 26);
        v65 = (unsigned __int8 *)*((_QWORD *)v96 + 12);
        if ( !v72 )
          goto LABEL_96;
        v66 = v72 - 1;
      }
      LODWORD(v67) = v66 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
      v68 = *(unsigned __int8 **)&v65[72 * (unsigned int)v67];
      if ( v46 == v68 )
        goto LABEL_71;
      v79 = 1;
      while ( v68 != (unsigned __int8 *)-4096LL )
      {
        v67 = v66 & (unsigned int)(v67 + v79);
        v68 = *(unsigned __int8 **)&v65[72 * v67];
        if ( v46 == v68 )
          goto LABEL_71;
        ++v79;
      }
LABEL_96:
      v91 = v37[3];
      if ( sub_98ED70(v46, *((_QWORD *)v96 + 416), 0, 0, 0)
        || *(_QWORD *)(v97 + 184) && sub_2B0E280(*((_QWORD *)v91 + 2), 0, v97) )
      {
LABEL_71:
        v37 += 3;
        goto LABEL_50;
      }
    }
    v37 += 4;
    if ( v42 != v37 )
      continue;
    break;
  }
  v40 = v94 - v37;
LABEL_123:
  if ( v40 == 2 )
  {
LABEL_129:
    if ( sub_2B3B9D0(&v96, *v37) )
      goto LABEL_50;
    ++v37;
    goto LABEL_131;
  }
  if ( v40 == 3 )
  {
    if ( sub_2B3B9D0(&v96, *v37) )
      goto LABEL_50;
    ++v37;
    goto LABEL_129;
  }
  if ( v40 != 1 )
  {
    v37 = v94;
    goto LABEL_50;
  }
LABEL_131:
  if ( !sub_2B3B9D0(&v96, *v37) )
    v37 = v94;
LABEL_50:
  v52 = v102;
  v53 = (unsigned int)v103;
  if ( v37 == &(*v28)[*((unsigned int *)v28 + 2)] )
  {
    v80 = &v102[v53];
    v81 = v29;
    v82 = v28;
    if ( &v102[v53] != v102 )
    {
      do
      {
        v83 = *v52;
        *(_DWORD *)(*v81 + 4 * v83) = -1;
        if ( (unsigned int)*(*v82)[v83] - 12 <= 1 )
          (*v82)[v83] = (unsigned __int8 *)sub_ACADE0(*(__int64 ***)a1[1]);
        ++v52;
      }
      while ( v80 != v52 );
    }
    *(_BYTE *)a1[4] = 1;
  }
  else
  {
    v54 = v37 - *v28;
    if ( &v102[v53] != v102 )
    {
      v95 = &v102[v53];
      do
      {
        v55 = *v52;
        *(_DWORD *)(*v29 + 4 * v55) = v54;
        if ( (_DWORD)v54 != (_DWORD)v55 )
          (*v28)[v55] = (unsigned __int8 *)sub_ACADE0(*(__int64 ***)a1[1]);
        ++v52;
      }
      while ( v95 != v52 );
    }
  }
LABEL_26:
  v30 = v99;
  v31 = 16LL * v101;
LABEL_27:
  sub_C7D6A0(v30, v31, 8);
  if ( v102 != (int *)v104 )
    _libc_free((unsigned __int64)v102);
}
