// Function: sub_1E497B0
// Address: 0x1e497b0
//
__int64 __fastcall sub_1E497B0(_QWORD *a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 (*v4)(); // rcx
  __int64 v5; // rax
  __int64 (*v6)(void); // rdx
  __int64 v7; // rax
  __int64 v8; // r15
  unsigned __int64 i; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 j; // rbx
  _DWORD *v14; // rax
  unsigned int v15; // ebx
  unsigned __int64 v16; // kr10_8
  _BYTE *v17; // r14
  __int64 v18; // r13
  __int64 v19; // r14
  __int64 v20; // r15
  size_t v21; // r13
  void *v22; // rax
  unsigned __int64 v23; // kr00_8
  void *v24; // rax
  unsigned int v25; // r8d
  unsigned int v26; // r10d
  unsigned int v27; // ecx
  void *v28; // rdi
  _BYTE *v29; // r13
  __int64 v30; // r14
  _BYTE *v31; // rsi
  __int64 v32; // rbx
  __int64 *v33; // r13
  __int64 v34; // rbx
  __int64 v35; // r12
  unsigned int v36; // r14d
  void *v37; // r13
  unsigned __int64 v38; // kr08_8
  _QWORD *v39; // r15
  __int64 v40; // rax
  __int64 *v41; // r15
  __int64 *v42; // r12
  unsigned int v43; // r14d
  unsigned int v44; // r13d
  __int64 v45; // rdi
  __int64 *v46; // r12
  __int64 v47; // rdi
  int v48; // r8d
  int v49; // r9d
  __int64 v50; // rax
  __int64 v51; // r15
  __int64 (*v52)(); // rax
  _BYTE *v53; // rax
  __int64 v54; // rcx
  __int64 v55; // rcx
  void *v56; // rdi
  __int64 *v57; // rbx
  __int64 *v58; // r12
  unsigned int v59; // r13d
  __int64 v60; // r14
  __int64 k; // r14
  __int64 v63; // r12
  __int64 v64; // rbx
  __int64 v65; // [rsp+10h] [rbp-170h]
  unsigned int v67; // [rsp+30h] [rbp-150h]
  __int64 v68; // [rsp+30h] [rbp-150h]
  void *v69; // [rsp+38h] [rbp-148h]
  _BYTE *v70; // [rsp+38h] [rbp-148h]
  __int64 *v71; // [rsp+38h] [rbp-148h]
  __int64 v72; // [rsp+40h] [rbp-140h]
  __int64 v73; // [rsp+48h] [rbp-138h]
  unsigned __int64 m; // [rsp+48h] [rbp-138h]
  int v75; // [rsp+48h] [rbp-138h]
  __int64 v76; // [rsp+58h] [rbp-128h] BYREF
  __int64 v77; // [rsp+60h] [rbp-120h]
  __int64 v78; // [rsp+68h] [rbp-118h] BYREF
  void *v79; // [rsp+70h] [rbp-110h]
  unsigned __int64 v80; // [rsp+78h] [rbp-108h]
  unsigned int v81; // [rsp+80h] [rbp-100h]
  __int64 v82; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v83; // [rsp+98h] [rbp-E8h]
  void *v84; // [rsp+A0h] [rbp-E0h]
  unsigned __int64 v85; // [rsp+A8h] [rbp-D8h]
  unsigned int v86; // [rsp+B0h] [rbp-D0h]
  __int64 *v87; // [rsp+C0h] [rbp-C0h] BYREF
  _BYTE *v88; // [rsp+C8h] [rbp-B8h]
  _BYTE *v89; // [rsp+D0h] [rbp-B0h]
  __int64 v90; // [rsp+D8h] [rbp-A8h]
  __int64 v91; // [rsp+E0h] [rbp-A0h]
  void *src; // [rsp+E8h] [rbp-98h]
  unsigned __int64 v93; // [rsp+F0h] [rbp-90h]
  unsigned int v94; // [rsp+F8h] [rbp-88h]
  __int64 *v95; // [rsp+100h] [rbp-80h] BYREF
  __int64 v96; // [rsp+108h] [rbp-78h]
  _BYTE v97[112]; // [rsp+110h] [rbp-70h] BYREF

  v1 = (__int64 *)v97;
  v96 = 0x800000000LL;
  v2 = a1[265];
  v95 = (__int64 *)v97;
  v3 = a1[2];
  v65 = **(_QWORD **)(v2 + 32);
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 944LL);
  v5 = 0;
  if ( v4 != sub_1E40480 )
  {
    v64 = ((__int64 (__fastcall *)(__int64, _QWORD))v4)(v3, *(_QWORD *)(a1[4] + 16LL));
    v1 = &v95[(unsigned int)v96];
    v5 = v64;
  }
  *v1 = v5;
  LODWORD(v96) = v96 + 1;
  v6 = *(__int64 (**)(void))(**(_QWORD **)(a1[4] + 16LL) + 128LL);
  v7 = 0;
  if ( v6 != sub_1D0B140 )
    v7 = v6();
  v77 = v7;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v8 = sub_1DD5D10(v65);
  for ( i = sub_1DD5EE0(v65); i != v8; v8 = *(_QWORD *)(v8 + 8) )
  {
    v10 = *(_QWORD *)(v77 + 72);
    v11 = *(_QWORD *)(v77 + 96) + 10LL * *(unsigned __int16 *)(*(_QWORD *)(v8 + 16) + 6LL);
    v12 = v10 + 16LL * *(unsigned __int16 *)(v11 + 2);
    for ( j = v10 + 16LL * *(unsigned __int16 *)(v11 + 4); j != v12; ++v14[1] )
    {
      while ( 1 )
      {
        LODWORD(v87) = *(_DWORD *)(v12 + 4);
        if ( (unsigned int)sub_39FAC40((unsigned int)v87) == 1 )
          break;
        v12 += 16;
        if ( j == v12 )
          goto LABEL_11;
      }
      v12 += 16;
      v14 = sub_1E49390((__int64)&v78, (int *)&v87);
    }
LABEL_11:
    if ( (*(_BYTE *)v8 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v8 + 46) & 8) != 0 )
        v8 = *(_QWORD *)(v8 + 8);
    }
  }
  v87 = 0;
  v88 = 0;
  v90 = v77;
  v89 = 0;
  v91 = 0;
  src = 0;
  v93 = 0;
  v94 = 0;
  j___libc_free_0(0);
  v94 = v81;
  if ( v81 )
  {
    src = (void *)sub_22077B0(8LL * v81);
    v93 = v80;
    memcpy(src, v79, 8LL * v94);
  }
  else
  {
    src = 0;
    v93 = 0;
  }
  v73 = v90;
  j___libc_free_0(0);
  v15 = v94;
  if ( v94 )
  {
    v69 = (void *)sub_22077B0(8LL * v94);
    v16 = v93;
    memcpy(v69, src, 8LL * v15);
  }
  else
  {
    v69 = 0;
    v16 = 0;
  }
  v17 = v88;
  v18 = (__int64)v87;
  j___libc_free_0(0);
  v19 = (__int64)&v17[-v18];
  if ( v19 > 8 )
  {
    v72 = v19 >> 3;
    for ( k = ((v19 >> 3) - 2) / 2; ; --k )
    {
      v63 = *(_QWORD *)(v18 + 8 * k);
      v83 = 0;
      v84 = 0;
      v82 = v73;
      v85 = 0;
      v86 = 0;
      j___libc_free_0(0);
      v86 = v15;
      if ( v15 )
      {
        v84 = (void *)sub_22077B0(8LL * v15);
        v85 = v16;
        memcpy(v84, v69, 8LL * v86);
      }
      else
      {
        v84 = 0;
        v85 = 0;
      }
      sub_1E47810(v18, k, v72, v63, &v82);
      j___libc_free_0(v84);
      if ( !k )
        break;
    }
  }
  j___libc_free_0(v69);
  j___libc_free_0(0);
  v20 = sub_1DD5D10(v65);
  for ( m = sub_1DD5EE0(v65); v20 != m; v20 = *(_QWORD *)(v20 + 8) )
  {
    while ( 1 )
    {
      v76 = v20;
      v31 = v88;
      if ( v88 == v89 )
      {
        sub_1DCC370((__int64)&v87, v88, &v76);
      }
      else
      {
        if ( v88 )
        {
          *(_QWORD *)v88 = v20;
          v31 = v88;
        }
        v88 = v31 + 8;
      }
      v32 = v90;
      j___libc_free_0(0);
      v27 = v94;
      if ( v94 )
      {
        v67 = v94;
        v21 = 8LL * v94;
        v22 = (void *)sub_22077B0(v21);
        v23 = v93;
        v24 = memcpy(v22, src, v21);
        v25 = HIDWORD(v23);
        v26 = v23;
        v27 = v67;
        v28 = v24;
      }
      else
      {
        v25 = 0;
        v26 = 0;
        v28 = 0;
      }
      v29 = v88;
      v30 = (__int64)v87;
      v84 = v28;
      v85 = __PAIR64__(v25, v26);
      v86 = v27;
      v82 = v32;
      v83 = 1;
      j___libc_free_0(0);
      sub_1E47570(v30, ((__int64)&v29[-v30] >> 3) - 1, 0, *((_QWORD *)v29 - 1), (__int64)&v82);
      j___libc_free_0(v84);
      j___libc_free_0(0);
      if ( !v20 )
        BUG();
      if ( (*(_BYTE *)v20 & 4) == 0 )
        break;
      v20 = *(_QWORD *)(v20 + 8);
      if ( v20 == m )
        goto LABEL_36;
    }
    while ( (*(_BYTE *)(v20 + 46) & 8) != 0 )
      v20 = *(_QWORD *)(v20 + 8);
  }
LABEL_36:
  v33 = (__int64 *)v88;
  while ( v87 != v33 )
  {
    v34 = *v87;
    v35 = v90;
    j___libc_free_0(0);
    v36 = v94;
    if ( v94 )
    {
      v37 = (void *)sub_22077B0(8LL * v94);
      v38 = v93;
      memcpy(v37, src, 8LL * v36);
    }
    else
    {
      v37 = 0;
      v38 = 0;
    }
    v39 = v87;
    if ( v88 - (_BYTE *)v87 > 8 )
    {
      v70 = v88;
      j___libc_free_0(0);
      v53 = v70;
      v71 = (__int64 *)(v70 - 8);
      v54 = *v71;
      *((_QWORD *)v53 - 1) = *v39;
      v68 = v54;
      v82 = v35;
      v83 = 0;
      v84 = 0;
      v85 = 0;
      v86 = 0;
      j___libc_free_0(0);
      v86 = v36;
      v55 = v68;
      if ( v36 )
      {
        v84 = (void *)sub_22077B0(8LL * v36);
        v85 = v38;
        memcpy(v84, v37, 8LL * v86);
        v55 = v68;
      }
      else
      {
        v84 = 0;
        v85 = 0;
      }
      sub_1E47810((__int64)v39, 0, v71 - v39, v55, &v82);
      j___libc_free_0(v84);
      v56 = v37;
      v37 = 0;
      j___libc_free_0(v56);
    }
    j___libc_free_0(v37);
    v33 = (__int64 *)(v88 - 8);
    v88 -= 8;
    if ( **(_WORD **)(v34 + 16) > 0xFu )
    {
      v40 = sub_1E45EB0((__int64)a1, v34);
      v41 = v95;
      v75 = *(unsigned __int16 *)(v40 + 226);
      v42 = &v95[(unsigned int)v96];
      if ( *(_WORD *)(v40 + 226) )
      {
        v43 = 0;
        v44 = 0;
        while ( 1 )
        {
          while ( v41 == v42 )
          {
            if ( v75 == ++v43 )
              goto LABEL_47;
          }
          v45 = *v41++;
          if ( (unsigned __int8)sub_20E8BB0(v45, v34) )
          {
            ++v44;
            if ( v75 == ++v43 )
              break;
          }
        }
LABEL_47:
        v46 = &v41[-v44];
        if ( !v44 )
          goto LABEL_53;
        do
        {
          v47 = *--v41;
          sub_20E8EF0(v47, v34);
        }
        while ( v41 != v46 );
        if ( v44 < v43 )
        {
LABEL_53:
          while ( 2 )
          {
            v51 = 0;
            v52 = *(__int64 (**)())(*(_QWORD *)a1[2] + 944LL);
            if ( v52 == sub_1E40480 )
            {
              sub_20E8EF0(0, v34);
              v50 = (unsigned int)v96;
              if ( (unsigned int)v96 >= HIDWORD(v96) )
                goto LABEL_55;
            }
            else
            {
              v51 = ((__int64 (__fastcall *)(_QWORD, _QWORD))v52)(a1[2], *(_QWORD *)(a1[4] + 16LL));
              sub_20E8EF0(v51, v34);
              v50 = (unsigned int)v96;
              if ( (unsigned int)v96 >= HIDWORD(v96) )
              {
LABEL_55:
                sub_16CD150((__int64)&v95, v97, 0, 8, v48, v49);
                v50 = (unsigned int)v96;
              }
            }
            ++v44;
            v95[v50] = v51;
            LODWORD(v96) = v96 + 1;
            if ( v43 <= v44 )
              goto LABEL_36;
            continue;
          }
        }
        goto LABEL_36;
      }
    }
  }
  v57 = v95;
  v58 = &v95[(unsigned int)v96];
  v59 = v96;
  if ( v95 != v58 )
  {
    do
    {
      v60 = *v57;
      if ( *v57 )
      {
        j___libc_free_0(*(_QWORD *)(v60 + 40));
        j_j___libc_free_0(v60, 64);
      }
      ++v57;
    }
    while ( v58 != v57 );
  }
  LODWORD(v96) = 0;
  j___libc_free_0(src);
  if ( v87 )
    j_j___libc_free_0(v87, v89 - (_BYTE *)v87);
  j___libc_free_0(v79);
  if ( v95 != (__int64 *)v97 )
    _libc_free((unsigned __int64)v95);
  return v59;
}
