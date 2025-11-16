// Function: sub_38EFA80
// Address: 0x38efa80
//
__int64 __fastcall sub_38EFA80(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // edx
  int *v5; // rbx
  int v6; // eax
  int v7; // eax
  unsigned __int64 v8; // rdx
  int v9; // eax
  unsigned __int64 v10; // r15
  __m128i v11; // xmm0
  bool v12; // cc
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  int *v16; // rax
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdx
  int v19; // eax
  unsigned __int64 v20; // r15
  __m128i v21; // xmm1
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  int *v25; // rax
  unsigned __int64 v26; // rdi
  __int64 *v27; // rdi
  __int64 v28; // r12
  __int64 v30; // rax
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rax
  unsigned __int64 v36; // rcx
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rdx
  _DWORD *v39; // rcx
  __int64 v40; // rax
  unsigned __int64 v41; // rcx
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rdx
  unsigned __int64 v47; // rcx
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rax
  _DWORD *v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rdx
  unsigned __int64 v53; // rcx
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rax
  unsigned int v58; // edx
  __int64 v59; // rbx
  __int64 v60; // rax
  __int64 v61; // rcx
  __int64 v62; // rdx
  _QWORD *v63; // rax
  unsigned __int64 v64; // r13
  const char *v65; // rbx
  __int64 v66; // r15
  unsigned __int64 v67; // r12
  unsigned __int64 v68; // rdi
  __int64 v69; // r12
  __int64 v70; // rax
  unsigned __int64 v71; // rax
  __int64 *v72; // rdi
  __int64 v73; // rsi
  int v74; // [rsp+Ch] [rbp-E4h]
  __int64 v75[2]; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v76; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v77; // [rsp+28h] [rbp-C8h]
  int v78; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v79; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v80; // [rsp+48h] [rbp-A8h] BYREF
  unsigned int v81; // [rsp+50h] [rbp-A0h]
  int v82; // [rsp+60h] [rbp-90h] BYREF
  __m128i v83; // [rsp+68h] [rbp-88h]
  unsigned __int64 v84; // [rsp+78h] [rbp-78h] BYREF
  unsigned int v85; // [rsp+80h] [rbp-70h]
  const char *v86; // [rsp+90h] [rbp-60h] BYREF
  unsigned __int64 v87; // [rsp+98h] [rbp-58h]
  __int64 v88; // [rsp+A0h] [rbp-50h]
  unsigned __int64 v89; // [rsp+A8h] [rbp-48h]
  unsigned int v90; // [rsp+B0h] [rbp-40h]

  v79 = 0u;
  v81 = 1;
  v80 = 0;
  v3 = sub_3909460(a1);
  v82 = *(_DWORD *)v3;
  v4 = *(_DWORD *)(v3 + 32);
  v83 = _mm_loadu_si128((const __m128i *)(v3 + 8));
  v85 = v4;
  if ( v4 > 0x40 )
    sub_16A4FD0((__int64)&v84, (const void **)(v3 + 24));
  else
    v84 = *(_QWORD *)(v3 + 24);
  v5 = *(int **)(a1 + 152);
  v74 = 0;
  v6 = *v5;
  if ( !*v5 )
  {
LABEL_36:
    v27 = *(__int64 **)(a1 + 344);
    v86 = "no matching '.endr' in definition";
    LOWORD(v88) = 259;
    *(_BYTE *)(a1 + 17) = 1;
    v76 = 0;
    v77 = 0;
    sub_16D14E0(v27, a2, 0, (__int64)&v86, (unsigned __int64 *)&v76, 1, 0, 0, 1u);
LABEL_37:
    v28 = 0;
    sub_38E35B0((_QWORD *)a1);
    goto LABEL_38;
  }
  while ( 1 )
  {
    if ( v6 != 2 )
      goto LABEL_5;
    v35 = sub_3909460(a1);
    if ( *(_DWORD *)v35 == 2 )
    {
      v39 = *(_DWORD **)(v35 + 8);
      v38 = *(_QWORD *)(v35 + 16);
    }
    else
    {
      v36 = *(_QWORD *)(v35 + 16);
      if ( !v36 )
        goto LABEL_67;
      v37 = v36 - 1;
      if ( v36 == 1 )
        v37 = 1;
      if ( v37 > v36 )
        v37 = *(_QWORD *)(v35 + 16);
      v38 = v37 - 1;
      v39 = (_DWORD *)(*(_QWORD *)(v35 + 8) + 1LL);
    }
    if ( v38 == 4 && *v39 == 1885696558 )
    {
LABEL_66:
      ++v74;
      v5 = *(int **)(a1 + 152);
      goto LABEL_5;
    }
LABEL_67:
    v40 = sub_3909460(a1);
    if ( *(_DWORD *)v40 == 2 )
    {
      v44 = *(_QWORD *)(v40 + 8);
      v43 = *(_QWORD *)(v40 + 16);
    }
    else
    {
      v41 = *(_QWORD *)(v40 + 16);
      if ( !v41 )
        goto LABEL_76;
      v42 = v41 - 1;
      if ( v41 == 1 )
        v42 = 1;
      if ( v42 > v41 )
        v42 = *(_QWORD *)(v40 + 16);
      v43 = v42 - 1;
      v44 = *(_QWORD *)(v40 + 8) + 1LL;
    }
    if ( v43 == 5 && *(_DWORD *)v44 == 1885696558 && *(_BYTE *)(v44 + 4) == 116 )
      goto LABEL_66;
LABEL_76:
    v45 = sub_3909460(a1);
    v46 = v45;
    if ( *(_DWORD *)v45 == 2 )
    {
      v50 = *(_DWORD **)(v45 + 8);
      v49 = *(_QWORD *)(v45 + 16);
    }
    else
    {
      v47 = *(_QWORD *)(v45 + 16);
      if ( !v47 )
        goto LABEL_85;
      v48 = v47 - 1;
      if ( v47 == 1 )
        v48 = 1;
      if ( v48 > v47 )
        v48 = v47;
      v49 = v48 - 1;
      v50 = (_DWORD *)(*(_QWORD *)(v46 + 8) + 1LL);
    }
    if ( v49 == 4 && *v50 == 1886546222 )
      goto LABEL_66;
LABEL_85:
    v51 = sub_3909460(a1);
    v52 = v51;
    if ( *(_DWORD *)v51 == 2 )
    {
      v56 = *(_QWORD *)(v51 + 8);
      v55 = *(_QWORD *)(v51 + 16);
    }
    else
    {
      v53 = *(_QWORD *)(v51 + 16);
      if ( !v53 )
        goto LABEL_94;
      v54 = v53 - 1;
      if ( v53 == 1 )
        v54 = 1;
      if ( v54 > v53 )
        v54 = v53;
      v55 = v54 - 1;
      v56 = *(_QWORD *)(v52 + 8) + 1LL;
    }
    if ( v55 != 5 || *(_DWORD *)v56 != 1886546222 )
    {
LABEL_94:
      v5 = *(int **)(a1 + 152);
      goto LABEL_5;
    }
    if ( *(_BYTE *)(v56 + 4) == 99 )
      goto LABEL_66;
    v5 = *(int **)(a1 + 152);
LABEL_5:
    v7 = *v5;
    if ( *v5 == 2 )
      break;
    if ( v7 != 9 )
    {
      do
      {
        if ( !v7 )
          goto LABEL_36;
        v8 = *(unsigned int *)(a1 + 160);
        *(_BYTE *)(a1 + 258) = 0;
        v9 = v8;
        v8 *= 40LL;
        v10 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v8 - 40) >> 3);
        if ( v8 > 0x28 )
        {
          do
          {
            v11 = _mm_loadu_si128((const __m128i *)v5 + 3);
            v12 = (unsigned int)v5[8] <= 0x40;
            *v5 = v5[10];
            *(__m128i *)(v5 + 2) = v11;
            if ( !v12 )
            {
              v13 = *((_QWORD *)v5 + 3);
              if ( v13 )
                j_j___libc_free_0_0(v13);
            }
            v14 = *((_QWORD *)v5 + 8);
            v5 += 10;
            *((_QWORD *)v5 - 2) = v14;
            LODWORD(v14) = v5[8];
            v5[8] = 0;
            *(v5 - 2) = v14;
            --v10;
          }
          while ( v10 );
          v9 = *(_DWORD *)(a1 + 160);
          v5 = *(int **)(a1 + 152);
        }
        v15 = (unsigned int)(v9 - 1);
        *(_DWORD *)(a1 + 160) = v15;
        v16 = &v5[10 * v15];
        if ( (unsigned int)v16[8] > 0x40 )
        {
          v17 = *((_QWORD *)v16 + 3);
          if ( v17 )
            j_j___libc_free_0_0(v17);
        }
        if ( !*(_DWORD *)(a1 + 160) )
        {
          sub_392C2E0(&v86, a1 + 144);
          sub_38E90E0(a1 + 152, *(_QWORD *)(a1 + 152), (unsigned __int64)&v86);
          if ( v90 > 0x40 )
          {
            if ( v89 )
              j_j___libc_free_0_0(v89);
          }
        }
LABEL_21:
        v5 = *(int **)(a1 + 152);
        v7 = *v5;
      }
      while ( *v5 != 9 );
    }
    v18 = *(unsigned int *)(a1 + 160);
    *(_BYTE *)(a1 + 258) = 1;
    v19 = v18;
    v18 *= 40LL;
    v20 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v18 - 40) >> 3);
    if ( v18 > 0x28 )
    {
      do
      {
        v21 = _mm_loadu_si128((const __m128i *)v5 + 3);
        v12 = (unsigned int)v5[8] <= 0x40;
        *v5 = v5[10];
        *(__m128i *)(v5 + 2) = v21;
        if ( !v12 )
        {
          v22 = *((_QWORD *)v5 + 3);
          if ( v22 )
            j_j___libc_free_0_0(v22);
        }
        v23 = *((_QWORD *)v5 + 8);
        v5 += 10;
        *((_QWORD *)v5 - 2) = v23;
        LODWORD(v23) = v5[8];
        v5[8] = 0;
        *(v5 - 2) = v23;
        --v20;
      }
      while ( v20 );
      v19 = *(_DWORD *)(a1 + 160);
      v5 = *(int **)(a1 + 152);
    }
    v24 = (unsigned int)(v19 - 1);
    *(_DWORD *)(a1 + 160) = v24;
    v25 = &v5[10 * v24];
    if ( (unsigned int)v25[8] > 0x40 )
    {
      v26 = *((_QWORD *)v25 + 3);
      if ( v26 )
        j_j___libc_free_0_0(v26);
    }
    if ( !*(_DWORD *)(a1 + 160) )
    {
      sub_392C2E0(&v86, a1 + 144);
      sub_38E90E0(a1 + 152, *(_QWORD *)(a1 + 152), (unsigned __int64)&v86);
      if ( v90 > 0x40 )
      {
        if ( v89 )
          j_j___libc_free_0_0(v89);
      }
    }
    v5 = *(int **)(a1 + 152);
    v6 = *v5;
    if ( !*v5 )
      goto LABEL_36;
  }
  v30 = sub_3909460(a1);
  if ( *(_DWORD *)v30 == 2 )
  {
    v34 = *(_QWORD *)(v30 + 8);
    v33 = *(_QWORD *)(v30 + 16);
  }
  else
  {
    v31 = *(_QWORD *)(v30 + 16);
    if ( !v31 )
      goto LABEL_21;
    v32 = v31 - 1;
    if ( v31 == 1 )
      v32 = 1;
    if ( v32 > v31 )
      v32 = *(_QWORD *)(v30 + 16);
    v33 = v32 - 1;
    v34 = *(_QWORD *)(v30 + 8) + 1LL;
  }
  if ( v33 != 5 || *(_DWORD *)v34 != 1684956462 || *(_BYTE *)(v34 + 4) != 114 )
    goto LABEL_21;
  if ( v74 )
  {
    --v74;
    goto LABEL_21;
  }
  v57 = sub_3909460(a1);
  v78 = *(_DWORD *)v57;
  v79 = _mm_loadu_si128((const __m128i *)(v57 + 8));
  if ( v81 <= 0x40 && (v58 = *(_DWORD *)(v57 + 32), v58 <= 0x40) )
  {
    v73 = *(_QWORD *)(v57 + 24);
    v81 = *(_DWORD *)(v57 + 32);
    v80 = v73 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v58);
  }
  else
  {
    sub_16A51C0((__int64)&v80, v57 + 24);
  }
  sub_38EB180(a1);
  if ( **(_DWORD **)(a1 + 152) != 9 )
  {
    v86 = "unexpected token in '.endr' directive";
    LOWORD(v88) = 259;
    v70 = sub_3909460(a1);
    v71 = sub_39092A0(v70);
    *(_BYTE *)(a1 + 17) = 1;
    v72 = *(__int64 **)(a1 + 344);
    v76 = 0;
    v77 = 0;
    sub_16D14E0(v72, v71, 0, (__int64)&v86, (unsigned __int64 *)&v76, 1, 0, 0, 1u);
    goto LABEL_37;
  }
  v59 = sub_39092A0(&v82);
  v60 = sub_39092A0(&v78);
  v61 = *(_QWORD *)(a1 + 536);
  v75[0] = v59;
  v86 = 0;
  v75[1] = v60 - v59;
  v62 = v60 - v59;
  v63 = *(_QWORD **)(a1 + 520);
  v87 = 0;
  v88 = 0;
  v76 = 0;
  v77 = 0;
  if ( v63 == (_QWORD *)(v61 - 56) )
  {
    sub_38E9C50((unsigned __int64 *)(a1 + 472), &v76, v75, (unsigned __int64 *)&v86);
    v64 = v87;
    v65 = v86;
LABEL_112:
    if ( v65 != (const char *)v64 )
    {
      do
      {
        v66 = *((_QWORD *)v65 + 3);
        v67 = *((_QWORD *)v65 + 2);
        if ( v66 != v67 )
        {
          do
          {
            if ( *(_DWORD *)(v67 + 32) > 0x40u )
            {
              v68 = *(_QWORD *)(v67 + 24);
              if ( v68 )
                j_j___libc_free_0_0(v68);
            }
            v67 += 40LL;
          }
          while ( v66 != v67 );
          v67 = *((_QWORD *)v65 + 2);
        }
        if ( v67 )
          j_j___libc_free_0(v67);
        v65 += 48;
      }
      while ( v65 != (const char *)v64 );
      v64 = (unsigned __int64)v86;
    }
    if ( v64 )
      j_j___libc_free_0(v64);
  }
  else
  {
    if ( v63 )
    {
      *v63 = 0;
      v63[1] = 0;
      v63[2] = v59;
      v63[3] = v62;
      v63[4] = 0;
      v63[5] = 0;
      v63[6] = 0;
      v64 = v87;
      *(_QWORD *)(a1 + 520) += 56LL;
      v65 = v86;
      goto LABEL_112;
    }
    *(_QWORD *)(a1 + 520) = 56;
  }
  v69 = *(_QWORD *)(a1 + 520);
  if ( v69 == *(_QWORD *)(a1 + 528) )
    v69 = *(_QWORD *)(*(_QWORD *)(a1 + 544) - 8LL) + 504LL;
  v28 = v69 - 56;
LABEL_38:
  if ( v85 > 0x40 && v84 )
    j_j___libc_free_0_0(v84);
  if ( v81 > 0x40 && v80 )
    j_j___libc_free_0_0(v80);
  return v28;
}
