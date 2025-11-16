// Function: sub_1A08FD0
// Address: 0x1a08fd0
//
void __fastcall sub_1A08FD0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v3; // r8
  int v4; // r14d
  unsigned int v5; // eax
  int v6; // r14d
  __int64 v7; // rbx
  __int64 *v8; // r10
  __int64 v9; // r15
  __int64 v10; // r9
  __int64 v11; // rsi
  __int64 *v12; // rdi
  __int64 *v13; // rax
  __int64 *v14; // rcx
  __int64 v15; // r15
  unsigned int v16; // r13d
  _QWORD *v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // rbx
  _BOOL4 v29; // eax
  int v30; // r9d
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 *v35; // rax
  __int64 *v36; // r8
  __int64 *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rcx
  char v40; // al
  int v41; // eax
  __int64 v42; // rdi
  __int64 v43; // r8
  __int64 v44; // r12
  __int64 v45; // r13
  char v46; // al
  int v47; // eax
  __int64 *v48; // r15
  __int64 *v49; // r12
  __int64 v50; // r14
  __int64 v51; // rax
  _QWORD *v52; // r13
  __int64 *v53; // rax
  __int64 v54; // rdx
  __int64 *v55; // rax
  __int64 v56; // rsi
  unsigned __int64 v57; // rcx
  __int64 v58; // rcx
  __int64 v59; // rbx
  __int64 v60; // r12
  unsigned __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rbx
  _BOOL4 v64; // eax
  int v65; // r9d
  __int64 v66; // rdx
  unsigned __int64 v67; // rcx
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rbx
  int v71; // r8d
  int v72; // r9d
  __int64 v73; // rdx
  unsigned __int64 v74; // rcx
  __int64 v75; // rdx
  __int64 v76; // rax
  unsigned __int64 v77; // rdx
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 *v81; // [rsp+20h] [rbp-130h]
  __int64 v82; // [rsp+28h] [rbp-128h]
  __int64 v83; // [rsp+28h] [rbp-128h]
  __int64 v84; // [rsp+28h] [rbp-128h]
  __int64 v85; // [rsp+28h] [rbp-128h]
  __int64 *v88; // [rsp+40h] [rbp-110h] BYREF
  __int64 v89; // [rsp+48h] [rbp-108h]
  _WORD v90[8]; // [rsp+50h] [rbp-100h] BYREF
  _BYTE *v91; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v92; // [rsp+68h] [rbp-E8h]
  _BYTE v93[64]; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v94; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 *v95; // [rsp+B8h] [rbp-98h]
  __int64 *v96; // [rsp+C0h] [rbp-90h]
  __int64 v97; // [rsp+C8h] [rbp-88h]
  int v98; // [rsp+D0h] [rbp-80h]
  _BYTE v99[120]; // [rsp+D8h] [rbp-78h] BYREF

  v3 = (__int64 *)v99;
  v4 = *(unsigned __int8 *)(a2 + 16);
  v91 = v93;
  v92 = 0x800000000LL;
  v5 = *((_DWORD *)a3 + 2);
  v6 = v4 - 24;
  v94 = 0;
  v95 = (__int64 *)v99;
  v96 = (__int64 *)v99;
  v97 = 8;
  v98 = 0;
  if ( !v5 )
  {
    v10 = *a3;
    goto LABEL_15;
  }
  v7 = 0;
  v8 = (__int64 *)v99;
  v9 = 16LL * v5;
  v10 = *a3;
  do
  {
LABEL_5:
    v11 = *(_QWORD *)(v10 + v7 + 8);
    if ( v3 != v8 )
    {
LABEL_3:
      sub_16CCBA0((__int64)&v94, v11);
      v8 = v96;
      v3 = v95;
      v10 = *a3;
      goto LABEL_4;
    }
    v12 = &v3[HIDWORD(v97)];
    if ( v12 == v3 )
    {
LABEL_121:
      if ( HIDWORD(v97) >= (unsigned int)v97 )
        goto LABEL_3;
      ++HIDWORD(v97);
      *v12 = v11;
      v3 = v95;
      ++v94;
      v8 = v96;
      v10 = *a3;
    }
    else
    {
      v13 = v3;
      v14 = 0;
      while ( v11 != *v13 )
      {
        if ( *v13 == -2 )
          v14 = v13;
        if ( v12 == ++v13 )
        {
          if ( !v14 )
            goto LABEL_121;
          v7 += 16;
          *v14 = v11;
          v8 = v96;
          --v98;
          v3 = v95;
          ++v94;
          v10 = *a3;
          if ( v7 != v9 )
            goto LABEL_5;
          goto LABEL_14;
        }
      }
    }
LABEL_4:
    v7 += 16;
  }
  while ( v7 != v9 );
LABEL_14:
  if ( *((_DWORD *)a3 + 2) == 2 )
  {
    v44 = *(_QWORD *)(v10 + 8);
    v45 = *(_QWORD *)(v10 + 24);
    v42 = *(_QWORD *)(a2 - 48);
    v43 = *(_QWORD *)(a2 - 24);
    if ( v44 == v42 && v45 == v43 )
      goto LABEL_84;
    v15 = a2;
    v17 = 0;
    goto LABEL_93;
  }
LABEL_15:
  v15 = a2;
  v16 = 0;
  v17 = 0;
  while ( 2 )
  {
    v23 = *(_QWORD *)(v15 - 24);
    v24 = 16LL * v16;
    v25 = *(_QWORD *)(v10 + v24 + 8);
    if ( v25 != v23 )
    {
      if ( v25 == *(_QWORD *)(v15 - 48) )
      {
        sub_15FB800(v15);
      }
      else
      {
        v82 = *(_QWORD *)(v10 + v24 + 8);
        v26 = sub_19FF050(v23, v6);
        v27 = v82;
        v28 = v26;
        if ( v26 )
        {
          v29 = sub_1A018F0((__int64)&v94, v26);
          v27 = v82;
          if ( !v29 )
          {
            v76 = (unsigned int)v92;
            if ( (unsigned int)v92 >= HIDWORD(v92) )
            {
              sub_16CD150((__int64)&v91, v93, 0, 8, v82, v30);
              v76 = (unsigned int)v92;
              v27 = v82;
            }
            *(_QWORD *)&v91[8 * v76] = v28;
            LODWORD(v92) = v92 + 1;
          }
        }
        if ( *(_QWORD *)(v15 - 24) )
        {
          v31 = *(_QWORD *)(v15 - 16);
          v32 = *(_QWORD *)(v15 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v32 = v31;
          if ( v31 )
            *(_QWORD *)(v31 + 16) = *(_QWORD *)(v31 + 16) & 3LL | v32;
        }
        *(_QWORD *)(v15 - 24) = v27;
        v17 = (_QWORD *)v15;
        if ( v27 )
        {
          v33 = *(_QWORD *)(v27 + 8);
          *(_QWORD *)(v15 - 16) = v33;
          if ( v33 )
            *(_QWORD *)(v33 + 16) = (v15 - 16) | *(_QWORD *)(v33 + 16) & 3LL;
          v17 = (_QWORD *)v15;
          *(_QWORD *)(v15 - 8) = (v27 + 8) | *(_QWORD *)(v15 - 8) & 3LL;
          *(_QWORD *)(v27 + 8) = v15 - 24;
        }
      }
      *(_BYTE *)(a1 + 752) = 1;
    }
    v34 = sub_19FF050(*(_QWORD *)(v15 - 48), v6);
    if ( !v34 )
      goto LABEL_43;
    v35 = v95;
    if ( v96 == v95 )
    {
      v36 = &v95[HIDWORD(v97)];
      if ( v95 == v36 )
      {
        v77 = (unsigned __int64)v95;
      }
      else
      {
        do
        {
          if ( v34 == *v35 )
            break;
          ++v35;
        }
        while ( v36 != v35 );
        v77 = (unsigned __int64)&v95[HIDWORD(v97)];
      }
    }
    else
    {
      v83 = v34;
      v81 = &v96[(unsigned int)v97];
      v35 = sub_16CC9F0((__int64)&v94, v34);
      v34 = v83;
      v36 = v81;
      if ( v83 == *v35 )
      {
        v77 = (unsigned __int64)(v96 == v95 ? &v96[HIDWORD(v97)] : &v96[(unsigned int)v97]);
      }
      else
      {
        if ( v96 != v95 )
        {
          v35 = &v96[(unsigned int)v97];
          goto LABEL_42;
        }
        v35 = &v96[HIDWORD(v97)];
        v77 = (unsigned __int64)v35;
      }
    }
    if ( v35 != (__int64 *)v77 )
    {
      while ( (unsigned __int64)*v35 >= 0xFFFFFFFFFFFFFFFELL )
      {
        if ( (__int64 *)v77 == ++v35 )
        {
          if ( v36 != v35 )
            goto LABEL_43;
          goto LABEL_59;
        }
      }
    }
LABEL_42:
    if ( v36 != v35 )
    {
LABEL_43:
      if ( (_DWORD)v92 )
      {
        v18 = *(_QWORD *)&v91[8 * (unsigned int)v92 - 8];
        LODWORD(v92) = v92 - 1;
        if ( *(_QWORD *)(v15 - 48) )
          goto LABEL_17;
      }
      else
      {
        v37 = (__int64 *)sub_1599EF0(*(__int64 ***)a2);
        v90[0] = 257;
        v38 = sub_15FB440(v6, v37, (__int64)v37, (__int64)&v88, a2);
        v39 = *(_QWORD *)v38;
        v18 = v38;
        v40 = *(_BYTE *)(*(_QWORD *)v38 + 8LL);
        if ( v40 == 16 )
          v40 = *(_BYTE *)(**(_QWORD **)(v39 + 16) + 8LL);
        if ( (unsigned __int8)(v40 - 1) <= 5u )
        {
          v84 = v18;
          v41 = sub_15F24E0(a2);
          sub_15F2440(v84, v41);
          v18 = v84;
        }
        if ( !*(_QWORD *)(v15 - 48) )
        {
          *(_QWORD *)(v15 - 48) = v18;
LABEL_20:
          v21 = *(_QWORD *)(v18 + 8);
          *(_QWORD *)(v15 - 40) = v21;
          if ( v21 )
            *(_QWORD *)(v21 + 16) = (v15 - 40) | *(_QWORD *)(v21 + 16) & 3LL;
          *(_QWORD *)(v15 - 32) = (v18 + 8) | *(_QWORD *)(v15 - 32) & 3LL;
          *(_QWORD *)(v18 + 8) = v15 - 48;
LABEL_23:
          v17 = (_QWORD *)v15;
          v15 = v18;
          *(_BYTE *)(a1 + 752) = 1;
          v22 = v16 + 1;
          if ( *((_DWORD *)a3 + 2) == v16 + 3 )
            goto LABEL_60;
LABEL_24:
          v10 = *a3;
          v16 = v22;
          continue;
        }
LABEL_17:
        v19 = *(_QWORD *)(v15 - 40);
        v20 = *(_QWORD *)(v15 - 32) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v20 = v19;
        if ( v19 )
          *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
      }
      *(_QWORD *)(v15 - 48) = v18;
      if ( v18 )
        goto LABEL_20;
      goto LABEL_23;
    }
    break;
  }
LABEL_59:
  v15 = v34;
  v22 = v16 + 1;
  if ( *((_DWORD *)a3 + 2) != v16 + 3 )
    goto LABEL_24;
LABEL_60:
  v42 = *(_QWORD *)(v15 - 48);
  v43 = *(_QWORD *)(v15 - 24);
  v44 = *(_QWORD *)(*a3 + 16 * v22 + 8);
  v45 = *(_QWORD *)(*a3 + 16LL * (v16 + 2) + 8);
  if ( v44 == v42 && v45 == v43 )
  {
    if ( v17 )
      goto LABEL_63;
    goto LABEL_84;
  }
LABEL_93:
  if ( v44 == v43 && v45 == v42 )
  {
    sub_15FB800(v15);
    *(_BYTE *)(a1 + 752) = 1;
    if ( v17 )
      goto LABEL_63;
  }
  else
  {
    if ( v44 != v42 )
    {
      v85 = v43;
      v62 = sub_19FF050(v42, v6);
      v43 = v85;
      v63 = v62;
      if ( v62 )
      {
        v64 = sub_1A018F0((__int64)&v94, v62);
        v43 = v85;
        if ( !v64 )
        {
          v79 = (unsigned int)v92;
          if ( (unsigned int)v92 >= HIDWORD(v92) )
          {
            sub_16CD150((__int64)&v91, v93, 0, 8, v85, v65);
            v79 = (unsigned int)v92;
            v43 = v85;
          }
          *(_QWORD *)&v91[8 * v79] = v63;
          LODWORD(v92) = v92 + 1;
        }
      }
      if ( *(_QWORD *)(v15 - 48) )
      {
        v66 = *(_QWORD *)(v15 - 40);
        v67 = *(_QWORD *)(v15 - 32) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v67 = v66;
        if ( v66 )
          *(_QWORD *)(v66 + 16) = v67 | *(_QWORD *)(v66 + 16) & 3LL;
      }
      *(_QWORD *)(v15 - 48) = v44;
      if ( v44 )
      {
        v68 = *(_QWORD *)(v44 + 8);
        *(_QWORD *)(v15 - 40) = v68;
        if ( v68 )
          *(_QWORD *)(v68 + 16) = (v15 - 40) | *(_QWORD *)(v68 + 16) & 3LL;
        *(_QWORD *)(v15 - 32) = (v44 + 8) | *(_QWORD *)(v15 - 32) & 3LL;
        *(_QWORD *)(v44 + 8) = v15 - 48;
      }
    }
    if ( v45 != v43 )
    {
      v69 = sub_19FF050(v43, v6);
      v70 = v69;
      if ( v69 && !sub_1A018F0((__int64)&v94, v69) )
      {
        v78 = (unsigned int)v92;
        if ( (unsigned int)v92 >= HIDWORD(v92) )
        {
          sub_16CD150((__int64)&v91, v93, 0, 8, v71, v72);
          v78 = (unsigned int)v92;
        }
        *(_QWORD *)&v91[8 * v78] = v70;
        LODWORD(v92) = v92 + 1;
      }
      if ( *(_QWORD *)(v15 - 24) )
      {
        v73 = *(_QWORD *)(v15 - 16);
        v74 = *(_QWORD *)(v15 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v74 = v73;
        if ( v73 )
          *(_QWORD *)(v73 + 16) = v74 | *(_QWORD *)(v73 + 16) & 3LL;
      }
      *(_QWORD *)(v15 - 24) = v45;
      if ( v45 )
      {
        v75 = *(_QWORD *)(v45 + 8);
        *(_QWORD *)(v15 - 16) = v75;
        if ( v75 )
          *(_QWORD *)(v75 + 16) = (v15 - 16) | *(_QWORD *)(v75 + 16) & 3LL;
        *(_QWORD *)(v15 - 8) = (v45 + 8) | *(_QWORD *)(v15 - 8) & 3LL;
        *(_QWORD *)(v45 + 8) = v15 - 24;
      }
    }
    v17 = (_QWORD *)v15;
    *(_BYTE *)(a1 + 752) = 1;
    while ( 1 )
    {
LABEL_63:
      v46 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
      if ( v46 == 16 )
        v46 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
      if ( (unsigned __int8)(v46 - 1) <= 5u || *(_BYTE *)(a2 + 16) == 76 )
      {
        v47 = sub_15F24E0(a2);
        *((_BYTE *)v17 + 17) &= 1u;
        sub_15F2440((__int64)v17, v47);
      }
      else
      {
        *((_BYTE *)v17 + 17) &= 1u;
      }
      if ( v17 == (_QWORD *)a2 )
        break;
      v88 = (__int64 *)v90;
      v89 = 0x100000000LL;
      sub_1AEA440(&v88, v17);
      v48 = v88;
      v49 = &v88[(unsigned int)v89];
      if ( v88 != v49 )
      {
        do
        {
          v50 = *v48;
          v51 = sub_1599EF0((__int64 **)*v17);
          v52 = sub_1624210(v51);
          v53 = (__int64 *)sub_16498A0(v50);
          v54 = sub_1628DA0(v53, (__int64)v52);
          v55 = (__int64 *)(v50 - 24LL * (*(_DWORD *)(v50 + 20) & 0xFFFFFFF));
          if ( *v55 )
          {
            v56 = v55[1];
            v57 = v55[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v57 = v56;
            if ( v56 )
              *(_QWORD *)(v56 + 16) = *(_QWORD *)(v56 + 16) & 3LL | v57;
          }
          *v55 = v54;
          if ( v54 )
          {
            v58 = *(_QWORD *)(v54 + 8);
            v55[1] = v58;
            if ( v58 )
              *(_QWORD *)(v58 + 16) = (unsigned __int64)(v55 + 1) | *(_QWORD *)(v58 + 16) & 3LL;
            v55[2] = (v54 + 8) | v55[2] & 3;
            *(_QWORD *)(v54 + 8) = v55;
          }
          ++v48;
        }
        while ( v49 != v48 );
      }
      sub_15F22F0(v17, a2);
      v17 = sub_1648700(v17[1]);
      if ( v88 != (__int64 *)v90 )
        _libc_free((unsigned __int64)v88);
    }
  }
LABEL_84:
  if ( (_DWORD)v92 )
  {
    v59 = 0;
    v60 = 8LL * (unsigned int)v92;
    do
    {
      v61 = *(_QWORD *)&v91[v59];
      v59 += 8;
      v88 = (__int64 *)v61;
      sub_1A062A0(a1 + 64, &v88);
    }
    while ( v60 != v59 );
  }
  if ( v96 != v95 )
    _libc_free((unsigned __int64)v96);
  if ( v91 != v93 )
    _libc_free((unsigned __int64)v91);
}
