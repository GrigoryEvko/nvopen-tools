// Function: sub_1F56E20
// Address: 0x1f56e20
//
__int64 __fastcall sub_1F56E20(
        __int64 a1,
        __m128 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 *v9; // rsi
  __int64 *v11; // rdi
  __int64 *v12; // rdx
  __int64 *v13; // rcx
  unsigned __int64 v14; // rbx
  __int64 v15; // rax
  __int64 *v16; // rax
  char v17; // si
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rbx
  __int64 v22; // r8
  char v23; // si
  __int64 *v24; // rax
  char v25; // si
  char v26; // r8
  __int64 v27; // rbx
  __int64 v28; // r13
  _QWORD *v29; // rax
  __int64 v30; // rdx
  __int64 *v31; // rsi
  __int64 v32; // r12
  __int64 v33; // rax
  double v34; // xmm4_8
  double v35; // xmm5_8
  _QWORD *v36; // r12
  unsigned __int64 *v37; // rcx
  unsigned __int64 v38; // rdx
  double v39; // xmm4_8
  double v40; // xmm5_8
  unsigned __int64 v41; // rax
  unsigned int v42; // r13d
  unsigned int v43; // esi
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rbx
  _BYTE *v48; // rsi
  __int64 v50; // r12
  __int64 v51; // [rsp+10h] [rbp-110h]
  __int64 v52; // [rsp+18h] [rbp-108h]
  int v53; // [rsp+18h] [rbp-108h]
  __int64 *v54; // [rsp+20h] [rbp-100h] BYREF
  __int64 *v55; // [rsp+28h] [rbp-F8h]
  __int64 *v56; // [rsp+30h] [rbp-F0h]
  char *v57; // [rsp+38h] [rbp-E8h]
  __int64 *v58; // [rsp+40h] [rbp-E0h] BYREF
  __int64 *v59; // [rsp+48h] [rbp-D8h]
  __int64 *v60; // [rsp+50h] [rbp-D0h]
  __int64 v61; // [rsp+58h] [rbp-C8h]
  __int64 v62; // [rsp+68h] [rbp-B8h]
  __int64 v63; // [rsp+70h] [rbp-B0h]
  __int64 v64; // [rsp+78h] [rbp-A8h]
  __int64 v65; // [rsp+80h] [rbp-A0h] BYREF
  _BYTE *v66; // [rsp+88h] [rbp-98h]
  _BYTE *v67; // [rsp+90h] [rbp-90h]
  __int64 v68; // [rsp+98h] [rbp-88h]
  int v69; // [rsp+A0h] [rbp-80h]
  _BYTE v70[120]; // [rsp+A8h] [rbp-78h] BYREF

  v9 = (__int64 *)&v54;
  v66 = v70;
  v67 = v70;
  v54 = (__int64 *)a1;
  v11 = (__int64 *)&v58;
  v65 = 0;
  v68 = 8;
  v69 = 0;
  sub_1CF17E0((__int64 *)&v58, (__int64 *)&v54, (__int64)&v65);
  v12 = v60;
  v55 = 0;
  v13 = v59;
  v56 = 0;
  v54 = v58;
  v57 = 0;
  v14 = (char *)v60 - (char *)v59;
  if ( v60 == v59 )
  {
    v11 = 0;
  }
  else
  {
    if ( v14 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_90;
    v15 = sub_22077B0((char *)v60 - (char *)v59);
    v12 = v60;
    v13 = v59;
    v11 = (__int64 *)v15;
  }
  v55 = v11;
  v56 = v11;
  v57 = (char *)v11 + v14;
  if ( v13 == v12 )
  {
    v12 = v11;
  }
  else
  {
    v16 = v11;
    v12 = (__int64 *)((char *)v11 + (char *)v12 - (char *)v13);
    do
    {
      if ( v16 )
      {
        *v16 = *v13;
        v17 = *((_BYTE *)v13 + 24);
        *((_BYTE *)v16 + 24) = v17;
        if ( v17 )
        {
          a2 = (__m128)_mm_loadu_si128((const __m128i *)(v13 + 1));
          *(__m128 *)(v16 + 1) = a2;
        }
      }
      v16 += 4;
      v13 += 4;
    }
    while ( v16 != v12 );
  }
  v18 = v62;
  v56 = v12;
  v9 = (__int64 *)(v63 - v62);
  v52 = v63 - v62;
  if ( v63 == v62 )
  {
    v20 = 0;
    goto LABEL_73;
  }
  if ( (unsigned __int64)v9 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_90:
    sub_4261EA(v11, v9, v12);
  v19 = sub_22077B0(v63 - v62);
  v18 = v62;
  v11 = v55;
  v12 = v56;
  v20 = v19;
  if ( v62 == v63 )
  {
LABEL_73:
    v21 = 0;
    goto LABEL_20;
  }
  v21 = v63 - v62;
  v22 = v19 + v63 - v62;
  do
  {
    if ( v19 )
    {
      *(_QWORD *)v19 = *(_QWORD *)v18;
      v23 = *(_BYTE *)(v18 + 24);
      *(_BYTE *)(v19 + 24) = v23;
      if ( v23 )
      {
        a3 = (__m128)_mm_loadu_si128((const __m128i *)(v18 + 8));
        *(__m128 *)(v19 + 8) = a3;
      }
    }
    v19 += 32;
    v18 += 32;
  }
  while ( v19 != v22 );
  if ( (char *)v12 - (char *)v11 == v21 )
    goto LABEL_21;
  while ( 1 )
  {
    do
    {
LABEL_19:
      sub_1F56CD0((__int64 *)&v54);
      v11 = v55;
      v12 = v56;
LABEL_20:
      ;
    }
    while ( (char *)v12 - (char *)v11 != v21 );
LABEL_21:
    if ( v11 == v12 )
      break;
    v18 = v20;
    v24 = v11;
    while ( *v24 == *(_QWORD *)v18 )
    {
      v25 = *((_BYTE *)v24 + 24);
      v26 = *(_BYTE *)(v18 + 24);
      if ( v25 && v26 )
      {
        if ( *((_DWORD *)v24 + 4) != *(_DWORD *)(v18 + 16) )
          goto LABEL_19;
        v24 += 4;
        v18 += 32;
        if ( v24 == v12 )
          goto LABEL_28;
      }
      else
      {
        if ( v25 != v26 )
          goto LABEL_19;
        v24 += 4;
        v18 += 32;
        if ( v24 == v12 )
          goto LABEL_28;
      }
    }
  }
LABEL_28:
  if ( v20 )
  {
    j_j___libc_free_0(v20, v52);
    v11 = v55;
  }
  if ( v11 )
    j_j___libc_free_0(v11, v57 - (char *)v11);
  if ( v62 )
    j_j___libc_free_0(v62, v64 - v62);
  if ( v59 )
    j_j___libc_free_0(v59, v61 - (_QWORD)v59);
  v27 = *(_QWORD *)(a1 + 80);
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v51 = a1 + 72;
  if ( a1 + 72 == v27 )
  {
    LODWORD(v50) = 0;
    goto LABEL_66;
  }
  do
  {
    while ( 1 )
    {
      v28 = v27 - 24;
      if ( !v27 )
        v28 = 0;
      v29 = v66;
      if ( v67 == v66 )
      {
        v50 = (__int64)&v66[8 * HIDWORD(v68)];
        if ( v66 == (_BYTE *)v50 )
        {
          v30 = (__int64)v66;
        }
        else
        {
          do
          {
            if ( v28 == *v29 )
              break;
            ++v29;
          }
          while ( (_QWORD *)v50 != v29 );
          v30 = (__int64)&v66[8 * HIDWORD(v68)];
        }
      }
      else
      {
        v50 = (__int64)&v67[8 * (unsigned int)v68];
        v29 = sub_16CC9F0((__int64)&v65, v28);
        if ( v28 == *v29 )
        {
          if ( v67 == v66 )
          {
            v18 = HIDWORD(v68);
            v30 = (__int64)&v67[8 * HIDWORD(v68)];
          }
          else
          {
            v18 = (unsigned int)v68;
            v30 = (__int64)&v67[8 * (unsigned int)v68];
          }
        }
        else
        {
          if ( v67 != v66 )
          {
            v30 = (unsigned int)v68;
            v29 = &v67[8 * (unsigned int)v68];
            goto LABEL_45;
          }
          v29 = &v67[8 * HIDWORD(v68)];
          v30 = (__int64)v29;
        }
      }
      while ( (_QWORD *)v30 != v29 && *v29 >= 0xFFFFFFFFFFFFFFFELL )
        ++v29;
LABEL_45:
      if ( v29 == (_QWORD *)v50 )
        break;
      v27 = *(_QWORD *)(v27 + 8);
      if ( v51 == v27 )
        goto LABEL_59;
    }
    v54 = (__int64 *)v28;
    v31 = v59;
    if ( v59 == v60 )
    {
      sub_1292090((__int64)&v58, v59, &v54);
      v28 = (__int64)v54;
    }
    else
    {
      if ( v59 )
      {
        *v59 = v28;
        v31 = v59;
      }
      v59 = ++v31;
    }
    v32 = *(_QWORD *)(v28 + 48);
    if ( !v32 )
      goto LABEL_92;
    if ( *(_BYTE *)(v32 - 8) == 77 )
    {
      while ( 1 )
      {
        v33 = sub_15A06D0(*(__int64 ***)(v32 - 24), (__int64)v31, v30, v18);
        sub_164D160(v32 - 24, v33, a2, *(double *)a3.m128_u64, a4, a5, v34, v35, a8, a9);
        v36 = (_QWORD *)v54[6];
        v31 = v36 - 3;
        sub_157EA20((__int64)(v54 + 5), (__int64)(v36 - 3));
        v37 = (unsigned __int64 *)v36[1];
        v38 = *v36 & 0xFFFFFFFFFFFFFFF8LL;
        *v37 = v38 | *v37 & 7;
        *(_QWORD *)(v38 + 8) = v37;
        *v36 &= 7uLL;
        v36[1] = 0;
        sub_164BEC0(
          (__int64)(v36 - 3),
          (__int64)(v36 - 3),
          v38,
          (__int64)v37,
          a2,
          *(double *)a3.m128_u64,
          a4,
          a5,
          v39,
          v40,
          a8,
          a9);
        v28 = (__int64)v54;
        v32 = v54[6];
        if ( !v32 )
          break;
        if ( *(_BYTE *)(v32 - 8) != 77 )
          goto LABEL_54;
      }
LABEL_92:
      BUG();
    }
LABEL_54:
    v41 = sub_157EBA0(v28);
    v50 = v41;
    if ( v41 )
    {
      v42 = 0;
      v53 = sub_15F4D60(v41);
      if ( v53 )
      {
        do
        {
          v43 = v42++;
          v44 = sub_15F4DF0(v50, v43);
          sub_157F2D0(v44, (__int64)v54, 0);
        }
        while ( v53 != v42 );
      }
      v28 = (__int64)v54;
    }
    sub_157EE90(v28);
    v27 = *(_QWORD *)(v27 + 8);
  }
  while ( v51 != v27 );
LABEL_59:
  v45 = (__int64)v58;
  v46 = v59 - v58;
  if ( (_DWORD)v46 )
  {
    v47 = 0;
    v50 = 8LL * (unsigned int)(v46 - 1);
    while ( 1 )
    {
      sub_157F980(*(_QWORD *)(v45 + v47));
      v45 = (__int64)v58;
      if ( v50 == v47 )
        break;
      v47 += 8;
    }
    v48 = (_BYTE *)((char *)v60 - (char *)v58);
    LOBYTE(v50) = v59 != v58;
  }
  else
  {
    LOBYTE(v50) = v59 != v58;
    v48 = (_BYTE *)((char *)v60 - (char *)v58);
  }
  if ( v45 )
    j_j___libc_free_0(v45, v48);
LABEL_66:
  if ( v67 != v66 )
    _libc_free((unsigned __int64)v67);
  return (unsigned int)v50;
}
