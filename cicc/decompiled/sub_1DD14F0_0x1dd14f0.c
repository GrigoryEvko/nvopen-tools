// Function: sub_1DD14F0
// Address: 0x1dd14f0
//
__int64 __fastcall sub_1DD14F0(__int64 a1, __int64 a2)
{
  __int64 (*v3)(); // rax
  __int64 v4; // rax
  unsigned __int64 v5; // r15
  __int64 v6; // r8
  int v7; // r9d
  __int64 v8; // r12
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  _QWORD *v12; // r15
  _QWORD *v13; // r12
  _QWORD *v14; // rdi
  __int64 v15; // rax
  __int64 *v16; // rdi
  __int64 *v17; // rsi
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r9
  unsigned __int64 v22; // r12
  __int64 v23; // rax
  __int64 v24; // rax
  char v25; // si
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 *v28; // rsi
  __int64 *v29; // r9
  __int64 v30; // rcx
  __int64 *v31; // rdx
  char v32; // r8
  unsigned __int64 v33; // r15
  __int64 v34; // rbx
  __int64 *v35; // rax
  char v36; // dl
  __int64 v37; // rdi
  __int64 v38; // r14
  __int64 *v39; // rax
  __int64 *v40; // rsi
  unsigned int v41; // r11d
  __int64 *v42; // r10
  __int64 v43; // rcx
  __int64 *v44; // rdx
  char v45; // si
  char v46; // r8
  unsigned int v47; // r12d
  __int64 v48; // rbx
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r15
  __int64 v53; // r14
  __int64 v54; // rax
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rdx
  __int64 v58; // rdi
  __int64 v59; // rax
  __int64 v60; // rax
  unsigned __int64 *v61; // r12
  unsigned __int64 *v62; // r14
  unsigned __int64 *v63; // rbx
  unsigned __int64 *v65; // r12
  unsigned __int64 *v66; // r15
  unsigned __int64 *v67; // rbx
  char *v68; // [rsp+8h] [rbp-1A8h]
  int v69; // [rsp+14h] [rbp-19Ch]
  unsigned __int64 v70; // [rsp+20h] [rbp-190h]
  int v71; // [rsp+28h] [rbp-188h]
  __int64 v72; // [rsp+30h] [rbp-180h]
  int v73; // [rsp+30h] [rbp-180h]
  unsigned __int64 v74; // [rsp+38h] [rbp-178h]
  __int64 v75; // [rsp+38h] [rbp-178h]
  __int64 v76; // [rsp+48h] [rbp-168h] BYREF
  __int64 v77[2]; // [rsp+50h] [rbp-160h] BYREF
  char v78; // [rsp+60h] [rbp-150h]
  __int64 v79; // [rsp+70h] [rbp-140h]
  __int64 *v80; // [rsp+78h] [rbp-138h] BYREF
  unsigned __int64 v81; // [rsp+80h] [rbp-130h]
  char *v82; // [rsp+88h] [rbp-128h]
  __int64 v83; // [rsp+90h] [rbp-120h] BYREF
  __int64 v84; // [rsp+98h] [rbp-118h]
  __int64 v85; // [rsp+A0h] [rbp-110h]
  __int64 v86; // [rsp+A8h] [rbp-108h]
  __int64 *v87; // [rsp+B8h] [rbp-F8h]
  __int64 *v88; // [rsp+C0h] [rbp-F0h]
  __int64 v89; // [rsp+C8h] [rbp-E8h]
  __int64 v90; // [rsp+D0h] [rbp-E0h] BYREF
  _BYTE *v91; // [rsp+D8h] [rbp-D8h]
  _BYTE *v92; // [rsp+E0h] [rbp-D0h]
  __int64 v93; // [rsp+E8h] [rbp-C8h]
  int v94; // [rsp+F0h] [rbp-C0h]
  _BYTE v95[184]; // [rsp+F8h] [rbp-B8h] BYREF

  *(_QWORD *)(a1 + 344) = a2;
  *(_QWORD *)(a1 + 352) = *(_QWORD *)(a2 + 40);
  v3 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 112LL);
  if ( v3 == sub_1D00B10 )
  {
    *(_QWORD *)(a1 + 360) = 0;
    BUG();
  }
  v4 = v3();
  *(_QWORD *)(a1 + 360) = v4;
  v5 = *(unsigned int *)(v4 + 16);
  v69 = *(_DWORD *)(v4 + 16);
  v90 = 0;
  v70 = v5;
  sub_1DCBF60(a1 + 368, v5, &v90);
  v90 = 0;
  sub_1DCBF60(a1 + 392, v5, &v90);
  v8 = *(_QWORD *)(a1 + 416);
  v9 = *(_QWORD *)(a1 + 424);
  v10 = (unsigned int)((__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 344) + 104LL) - *(_QWORD *)(*(_QWORD *)(a1 + 344) + 96LL)) >> 3);
  v11 = (v9 - v8) >> 5;
  if ( v10 > v11 )
  {
    sub_1DCC0D0((__int64 *)(a1 + 416), v10 - v11, v9, v11, v6, v7);
  }
  else if ( v10 < v11 )
  {
    v65 = (unsigned __int64 *)(32 * v10 + v8);
    if ( (unsigned __int64 *)v9 != v65 )
    {
      v66 = v65;
      v67 = *(unsigned __int64 **)(a1 + 424);
      do
      {
        if ( (unsigned __int64 *)*v66 != v66 + 2 )
          _libc_free(*v66);
        v66 += 4;
      }
      while ( v67 != v66 );
      *(_QWORD *)(a1 + 424) = v65;
    }
  }
  v12 = *(_QWORD **)(a1 + 320);
  v13 = (_QWORD *)(a1 + 320);
  while ( v12 != v13 )
  {
    v14 = v12;
    v12 = (_QWORD *)*v12;
    j_j___libc_free_0(v14, 40);
  }
  v15 = *(_QWORD *)(a1 + 352);
  *(_QWORD *)(a1 + 328) = v13;
  *(_QWORD *)(a1 + 320) = v13;
  *(_QWORD *)(a1 + 336) = 0;
  if ( (**(_BYTE **)(*(_QWORD *)v15 + 352LL) & 1) == 0 )
    sub_16BD130("regalloc=... not currently supported with -O0", 1u);
  sub_1DCB4F0(a1, a2);
  v16 = &v83;
  v17 = &v76;
  v18 = *(_QWORD *)(*(_QWORD *)(a1 + 344) + 328LL);
  v90 = 0;
  v93 = 16;
  v76 = v18;
  v91 = v95;
  v92 = v95;
  v94 = 0;
  sub_1DCD230(&v83, &v76, (__int64)&v90);
  v20 = v85;
  v80 = 0;
  v21 = v84;
  v81 = 0;
  v79 = v83;
  v82 = 0;
  v22 = v85 - v84;
  if ( v85 == v84 )
  {
    v16 = 0;
  }
  else
  {
    if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_99;
    v23 = sub_22077B0(v85 - v84);
    v20 = v85;
    v21 = v84;
    v16 = (__int64 *)v23;
  }
  v80 = v16;
  v81 = (unsigned __int64)v16;
  v82 = (char *)v16 + v22;
  if ( v21 == v20 )
  {
    v26 = (unsigned __int64)v16;
  }
  else
  {
    v19 = v16;
    v24 = v21;
    do
    {
      if ( v19 )
      {
        *v19 = *(_QWORD *)v24;
        v25 = *(_BYTE *)(v24 + 16);
        *((_BYTE *)v19 + 16) = v25;
        if ( v25 )
          v19[1] = *(_QWORD *)(v24 + 8);
      }
      v24 += 24;
      v19 += 3;
    }
    while ( v24 != v20 );
    v26 = (unsigned __int64)&v16[((unsigned __int64)(v24 - 24 - v21) >> 3) + 3];
  }
  v17 = v88;
  v81 = v26;
  v68 = (char *)((char *)v88 - (char *)v87);
  if ( v88 == v87 )
  {
    v72 = 0;
    goto LABEL_94;
  }
  if ( (unsigned __int64)((char *)v88 - (char *)v87) > 0x7FFFFFFFFFFFFFF8LL )
LABEL_99:
    sub_4261EA(v16, v17, v19);
  v27 = sub_22077B0((char *)v88 - (char *)v87);
  v28 = v88;
  v29 = v87;
  v72 = v27;
  v16 = v80;
  v26 = v81;
  if ( v88 == v87 )
  {
LABEL_94:
    v74 = 0;
    goto LABEL_26;
  }
  v30 = v72;
  v31 = v87;
  do
  {
    if ( v30 )
    {
      *(_QWORD *)v30 = *v31;
      v32 = *((_BYTE *)v31 + 16);
      *(_BYTE *)(v30 + 16) = v32;
      if ( v32 )
        *(_QWORD *)(v30 + 8) = v31[1];
    }
    v31 += 3;
    v30 += 24;
  }
  while ( v31 != v28 );
  v74 = 8 * ((unsigned __int64)((char *)(v31 - 3) - (char *)v29) >> 3) + 24;
  while ( 1 )
  {
LABEL_26:
    if ( v26 - (_QWORD)v16 != v74 )
      goto LABEL_27;
LABEL_42:
    if ( (__int64 *)v26 == v16 )
      break;
    v43 = v72;
    v44 = v16;
    while ( *v44 == *(_QWORD *)v43 )
    {
      v45 = *((_BYTE *)v44 + 16);
      v46 = *(_BYTE *)(v43 + 16);
      if ( v45 && v46 )
      {
        if ( v44[1] != *(_QWORD *)(v43 + 8) )
          break;
        v44 += 3;
        v43 += 24;
        if ( v44 == (__int64 *)v26 )
          goto LABEL_49;
      }
      else
      {
        if ( v46 != v45 )
          break;
        v44 += 3;
        v43 += 24;
        if ( v44 == (__int64 *)v26 )
          goto LABEL_49;
      }
    }
LABEL_27:
    sub_1DD0CF0(a1, *(_QWORD *)(v26 - 24), v69);
    v77[0] = 0;
    sub_1DCBF60(a1 + 368, v70, v77);
    v77[0] = 0;
    sub_1DCBF60(a1 + 392, v70, v77);
    v33 = v81;
    while ( 2 )
    {
      v34 = *(_QWORD *)(v33 - 24);
      if ( !*(_BYTE *)(v33 - 8) )
      {
        v35 = *(__int64 **)(v34 + 88);
        *(_BYTE *)(v33 - 8) = 1;
        *(_QWORD *)(v33 - 16) = v35;
        goto LABEL_32;
      }
      while ( 1 )
      {
        v35 = *(__int64 **)(v33 - 16);
LABEL_32:
        if ( *(__int64 **)(v34 + 96) == v35 )
          break;
        *(_QWORD *)(v33 - 16) = v35 + 1;
        v37 = v79;
        v38 = *v35;
        v39 = *(__int64 **)(v79 + 8);
        if ( *(__int64 **)(v79 + 16) != v39 )
          goto LABEL_30;
        v40 = &v39[*(unsigned int *)(v79 + 28)];
        v41 = *(_DWORD *)(v79 + 28);
        if ( v39 == v40 )
        {
LABEL_80:
          if ( v41 < *(_DWORD *)(v79 + 24) )
          {
            *(_DWORD *)(v79 + 28) = v41 + 1;
            *v40 = v38;
            ++*(_QWORD *)v37;
LABEL_41:
            v77[0] = v38;
            v78 = 0;
            sub_1BFDD10((unsigned __int64 *)&v80, (__int64)v77);
            v26 = v81;
            v16 = v80;
            if ( v81 - (_QWORD)v80 != v74 )
              goto LABEL_27;
            goto LABEL_42;
          }
LABEL_30:
          sub_16CCBA0(v79, v38);
          if ( v36 )
            goto LABEL_41;
        }
        else
        {
          v42 = 0;
          while ( v38 != *v39 )
          {
            if ( *v39 == -2 )
            {
              v42 = v39;
              if ( v40 == v39 + 1 )
                goto LABEL_40;
              ++v39;
            }
            else if ( v40 == ++v39 )
            {
              if ( !v42 )
                goto LABEL_80;
LABEL_40:
              *v42 = v38;
              --*(_DWORD *)(v37 + 32);
              ++*(_QWORD *)v37;
              goto LABEL_41;
            }
          }
        }
      }
      v81 -= 24LL;
      v26 = (unsigned __int64)v80;
      v33 = v81;
      if ( (__int64 *)v81 != v80 )
        continue;
      break;
    }
    v16 = v80;
  }
LABEL_49:
  if ( v72 )
  {
    j_j___libc_free_0(v72, v68);
    v16 = v80;
  }
  if ( v16 )
    j_j___libc_free_0(v16, v82 - (char *)v16);
  if ( v87 )
    j_j___libc_free_0(v87, v89 - (_QWORD)v87);
  if ( v84 )
    j_j___libc_free_0(v84, v86 - v84);
  v73 = 0;
  v71 = *(_DWORD *)(a1 + 240);
  if ( v71 )
  {
    do
    {
      v47 = v73 | 0x80000000;
      v48 = 56LL * (v73 & 0x7FFFFFFF);
      v49 = v48 + *(_QWORD *)(a1 + 232);
      v50 = *(_QWORD *)(v49 + 32);
      v51 = (*(_QWORD *)(v49 + 40) - v50) >> 3;
      if ( (_DWORD)v51 )
      {
        v52 = 0;
        v75 = 8LL * (unsigned int)(v51 - 1);
        while ( 1 )
        {
          v53 = *(_QWORD *)(v50 + v52);
          v54 = sub_1E69D00(*(_QWORD *)(a1 + 352), v47);
          v57 = *(_QWORD *)(a1 + 360);
          v58 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 232) + v48 + 32) + v52);
          if ( v53 == v54 )
          {
            sub_1E1B440(v58, v47, v57, 0);
            if ( v52 == v75 )
              break;
          }
          else
          {
            sub_1E1AFE0(v58, v47, v57, 0, v55, v56);
            if ( v52 == v75 )
              break;
          }
          v52 += 8;
          v50 = *(_QWORD *)(*(_QWORD *)(a1 + 232) + v48 + 32);
        }
      }
      ++v73;
    }
    while ( v73 != v71 );
  }
  v59 = *(_QWORD *)(a1 + 368);
  if ( v59 != *(_QWORD *)(a1 + 376) )
    *(_QWORD *)(a1 + 376) = v59;
  v60 = *(_QWORD *)(a1 + 392);
  if ( v60 != *(_QWORD *)(a1 + 400) )
    *(_QWORD *)(a1 + 400) = v60;
  v61 = *(unsigned __int64 **)(a1 + 416);
  v62 = *(unsigned __int64 **)(a1 + 424);
  if ( v61 != v62 )
  {
    v63 = *(unsigned __int64 **)(a1 + 416);
    do
    {
      if ( (unsigned __int64 *)*v63 != v63 + 2 )
        _libc_free(*v63);
      v63 += 4;
    }
    while ( v62 != v63 );
    *(_QWORD *)(a1 + 424) = v61;
  }
  if ( v92 != v91 )
    _libc_free((unsigned __int64)v92);
  return 0;
}
