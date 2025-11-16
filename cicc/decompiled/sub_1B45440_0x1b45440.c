// Function: sub_1B45440
// Address: 0x1b45440
//
__int64 __fastcall sub_1B45440(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  void **p_base; // r14
  unsigned __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // r11
  char *v10; // rax
  __int64 v11; // rdx
  char *v12; // rsi
  __int64 v13; // rdx
  __m128i *v14; // rax
  __int64 v15; // rsi
  __m128i *v16; // rsi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // r14
  __int64 v19; // rbx
  unsigned int i; // r15d
  __int64 v21; // rdi
  __m128i *v22; // rdi
  unsigned int v23; // r12d
  __m128i *v25; // rax
  void **v26; // rbx
  __m128i *v27; // r8
  __int64 v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rcx
  size_t v31; // rsi
  _QWORD *v32; // rax
  char *v33; // rax
  __int64 *v34; // r9
  __int64 v35; // rdx
  __int64 v36; // r14
  __int64 *v37; // r8
  __int64 v38; // rbx
  __int64 v39; // rsi
  __int64 *v40; // rdi
  __int64 *v41; // rax
  __int64 *v42; // rcx
  _QWORD *v43; // r8
  int v44; // r10d
  int v45; // r9d
  unsigned int v46; // esi
  unsigned int v47; // ecx
  unsigned __int64 v48; // rax
  bool v49; // zf
  __int64 v50; // rax
  int v51; // r9d
  __int64 v52; // r14
  __int64 v53; // rdx
  unsigned int v54; // eax
  __int64 v55; // r15
  unsigned __int64 v56; // rsi
  __int64 v57; // rdx
  __int64 v58; // rbx
  __int64 *v59; // rax
  __int64 *v60; // r13
  unsigned int *v61; // rax
  unsigned int *v62; // rdx
  unsigned int v63; // ecx
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // r8
  __int64 v68; // r13
  __int64 v69; // rdx
  _QWORD *v70; // rbx
  __int64 *v71; // rdx
  __int64 v72; // rdi
  __int64 v74; // [rsp+10h] [rbp-160h]
  char v75; // [rsp+10h] [rbp-160h]
  __int64 v76; // [rsp+18h] [rbp-158h]
  int v77; // [rsp+18h] [rbp-158h]
  __int64 v78; // [rsp+18h] [rbp-158h]
  __int64 v79; // [rsp+18h] [rbp-158h]
  void *base; // [rsp+20h] [rbp-150h] BYREF
  __m128i *v81; // [rsp+28h] [rbp-148h]
  __int64 v82; // [rsp+30h] [rbp-140h]
  __m128i *v83; // [rsp+40h] [rbp-130h] BYREF
  __m128i *v84; // [rsp+48h] [rbp-128h]
  __int64 v85; // [rsp+50h] [rbp-120h]
  unsigned int *v86; // [rsp+60h] [rbp-110h] BYREF
  __int64 v87; // [rsp+68h] [rbp-108h]
  _BYTE v88[32]; // [rsp+70h] [rbp-100h] BYREF
  __int64 v89; // [rsp+90h] [rbp-E0h] BYREF
  __int64 *v90; // [rsp+98h] [rbp-D8h]
  __int64 *v91; // [rsp+A0h] [rbp-D0h]
  __int64 v92; // [rsp+A8h] [rbp-C8h]
  int v93; // [rsp+B0h] [rbp-C0h]
  _BYTE v94[184]; // [rsp+B8h] [rbp-B8h] BYREF

  p_base = &base;
  base = 0;
  v81 = 0;
  v82 = 0;
  v6 = sub_157EBA0(a3);
  v7 = sub_1B449D0(a1, v6, (__int64)&base);
  sub_1B427F0(v7, (__m128i **)&base);
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v76 = sub_1B449D0(a1, a2, (__int64)&v83);
  sub_1B427F0(v76, &v83);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = v76;
  if ( v7 != v8 )
  {
    v10 = (char *)base;
    v11 = ((char *)v81 - (_BYTE *)base) >> 4;
    if ( (_DWORD)v11 )
    {
      v12 = (char *)base + 16 * (unsigned int)(v11 - 1) + 16;
      v13 = 0;
      while ( 1 )
      {
        while ( v8 != *((_QWORD *)v10 + 1) )
        {
          v10 += 16;
          if ( v12 == v10 )
            goto LABEL_8;
        }
        if ( v13 )
          goto LABEL_28;
        v13 = *(_QWORD *)v10;
        v10 += 16;
        if ( v12 == v10 )
          goto LABEL_8;
      }
    }
    v13 = 0;
LABEL_8:
    v14 = v83;
    v15 = v84 - v83;
    if ( !(_DWORD)v15 )
      goto LABEL_26;
    v16 = &v83[(unsigned int)(v15 - 1) + 1];
    while ( v14->m128i_i64[0] != v13 )
    {
      if ( v16 == ++v14 )
        goto LABEL_26;
    }
    v74 = v14->m128i_i64[1];
    if ( !v74 )
LABEL_26:
      v74 = v76;
    v17 = sub_157EBA0(v8);
    if ( v17 )
    {
      v77 = sub_15F4D60(v17);
      v18 = sub_157EBA0(v8);
      if ( v77 )
      {
        v19 = v74;
        for ( i = 0; i != v77; ++i )
        {
          while ( 1 )
          {
            v21 = sub_15F4DF0(v18, i);
            if ( v21 == v19 )
              break;
            ++i;
            sub_157F2D0(v21, v8, 0);
            if ( v77 == i )
              goto LABEL_19;
          }
          v19 = 0;
        }
      }
    }
LABEL_19:
    sub_1B44660(a4, v74);
    sub_1B44FE0(a2);
LABEL_20:
    v22 = v83;
    v23 = 1;
    goto LABEL_21;
  }
  v25 = v81;
  v26 = (void **)&v83;
  v27 = (__m128i *)base;
  v22 = v83;
  if ( (char *)v81 - (_BYTE *)base > (unsigned __int64)((char *)v84 - (char *)v83) )
  {
    v26 = &base;
    v27 = v83;
    v25 = v84;
    p_base = (void **)&v83;
  }
  if ( v27 == v25 )
    goto LABEL_29;
  v28 = (char *)v25 - (char *)v27;
  if ( v28 != 16 )
  {
    if ( v28 <= 16 )
    {
      v29 = *v26;
    }
    else
    {
      qsort(v27, v28 >> 4, 0x10u, (__compar_fn_t)sub_1B42380);
      v29 = *v26;
      v9 = v76;
    }
    v30 = (_BYTE *)v26[1] - (_BYTE *)*v26;
    v31 = v30 >> 4;
LABEL_56:
    if ( v30 <= 16 )
    {
LABEL_58:
      v43 = *p_base;
      v44 = v31;
      v45 = ((_BYTE *)p_base[1] - (_BYTE *)*p_base) >> 4;
      if ( v45 && (_DWORD)v31 )
      {
        v46 = 0;
        v47 = 0;
        do
        {
          v48 = v29[2 * v46];
          if ( v43[2 * v47] == v48 )
            goto LABEL_38;
          if ( v43[2 * v47] < v48 )
            ++v47;
          else
            ++v46;
        }
        while ( v45 != v47 && v44 != v46 );
      }
LABEL_28:
      v22 = v83;
LABEL_29:
      v23 = 0;
      goto LABEL_21;
    }
LABEL_57:
    v78 = v9;
    qsort(v29, v31, 0x10u, (__compar_fn_t)sub_1B42380);
    v29 = *v26;
    v9 = v78;
    v31 = ((_BYTE *)v26[1] - (_BYTE *)*v26) >> 4;
    goto LABEL_58;
  }
  v29 = *v26;
  v30 = (_BYTE *)v26[1] - (_BYTE *)*v26;
  v31 = v30 >> 4;
  if ( !(unsigned int)(v30 >> 4) )
  {
    if ( v30 <= 16 )
      goto LABEL_28;
    goto LABEL_57;
  }
  v32 = *v26;
  while ( v27->m128i_i64[0] != *v32 )
  {
    v32 += 2;
    if ( v32 == &v29[2 * (unsigned int)(v31 - 1) + 2] )
      goto LABEL_56;
  }
LABEL_38:
  if ( *(_BYTE *)(a2 + 16) != 26 )
  {
    v33 = (char *)base;
    v34 = (__int64 *)v94;
    v89 = 0;
    v90 = (__int64 *)v94;
    v91 = (__int64 *)v94;
    v92 = 16;
    v35 = ((char *)v81 - (_BYTE *)base) >> 4;
    v93 = 0;
    if ( (_DWORD)v35 )
    {
      v36 = 0;
      v37 = (__int64 *)v94;
      v38 = 16LL * (unsigned int)(v35 - 1);
      while ( 1 )
      {
        v39 = *(_QWORD *)&v33[v36];
        if ( v37 != v34 )
          break;
        v40 = &v37[HIDWORD(v92)];
        if ( v40 == v37 )
        {
LABEL_108:
          if ( HIDWORD(v92) >= (unsigned int)v92 )
            break;
          ++HIDWORD(v92);
          *v40 = v39;
          v34 = v90;
          ++v89;
          v37 = v91;
        }
        else
        {
          v41 = v37;
          v42 = 0;
          while ( v39 != *v41 )
          {
            if ( *v41 == -2 )
              v42 = v41;
            if ( v40 == ++v41 )
            {
              if ( !v42 )
                goto LABEL_108;
              *v42 = v39;
              v37 = v91;
              --v93;
              v34 = v90;
              ++v89;
              break;
            }
          }
        }
LABEL_42:
        if ( v36 == v38 )
          goto LABEL_67;
        v33 = (char *)base;
        v36 += 16;
      }
      sub_16CCBA0((__int64)&v89, v39);
      v37 = v91;
      v34 = v90;
      goto LABEL_42;
    }
LABEL_67:
    v49 = *(_QWORD *)(a2 + 48) == 0;
    v86 = (unsigned int *)v88;
    v87 = 0x800000000LL;
    if ( v49 && *(__int16 *)(a2 + 18) >= 0 || (v50 = sub_1625790(a2, 2), (v52 = v50) == 0) )
    {
      v75 = 0;
      v54 = (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1;
    }
    else
    {
      v53 = *(unsigned int *)(v50 + 8);
      v75 = 0;
      v54 = (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1;
      if ( (_DWORD)v53 == v54 + 1 )
      {
        if ( (_DWORD)v53 == 1 )
        {
          v75 = 1;
        }
        else
        {
          v66 = (unsigned int)v87;
          v67 = v53;
          v68 = 1;
          while ( 1 )
          {
            v69 = *(_QWORD *)(*(_QWORD *)(v52 + 8 * (v68 - v53)) + 136LL);
            v70 = *(_QWORD **)(v69 + 24);
            if ( *(_DWORD *)(v69 + 32) > 0x40u )
              v70 = (_QWORD *)*v70;
            if ( HIDWORD(v87) <= (unsigned int)v66 )
            {
              v79 = v67;
              sub_16CD150((__int64)&v86, v88, 0, 4, v67, v51);
              v66 = (unsigned int)v87;
              v67 = v79;
            }
            ++v68;
            v86[v66] = (unsigned int)v70;
            v66 = (unsigned int)(v87 + 1);
            LODWORD(v87) = v87 + 1;
            if ( v67 == v68 )
              break;
            v53 = *(unsigned int *)(v52 + 8);
          }
          v75 = 1;
          v54 = (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1;
        }
      }
    }
    v55 = v54 - 1;
LABEL_71:
    if ( !v55 )
    {
LABEL_88:
      if ( v75 && (unsigned int)v87 > 1uLL )
        sub_1B42940(a2, v86, (unsigned int)v87);
      if ( v86 != (unsigned int *)v88 )
        _libc_free((unsigned __int64)v86);
      if ( v91 != v90 )
        _libc_free((unsigned __int64)v91);
      goto LABEL_20;
    }
    v56 = (unsigned __int64)v91;
    while ( 1 )
    {
      --v55;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v57 = *(_QWORD *)(a2 - 8);
      else
        v57 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v58 = *(_QWORD *)(v57 + 24LL * (unsigned int)(2 * v55 + 2));
      v59 = v90;
      if ( (__int64 *)v56 == v90 )
      {
        v60 = (__int64 *)(v56 + 8LL * HIDWORD(v92));
        if ( (__int64 *)v56 == v60 )
        {
          v71 = (__int64 *)v56;
        }
        else
        {
          do
          {
            if ( v58 == *v59 )
              break;
            ++v59;
          }
          while ( v60 != v59 );
          v71 = (__int64 *)(v56 + 8LL * HIDWORD(v92));
        }
      }
      else
      {
        v60 = (__int64 *)(v56 + 8LL * (unsigned int)v92);
        v59 = sub_16CC9F0((__int64)&v89, *(_QWORD *)(v57 + 24LL * (unsigned int)(2 * v55 + 2)));
        if ( v58 == *v59 )
        {
          v56 = (unsigned __int64)v91;
          if ( v91 == v90 )
            v71 = &v91[HIDWORD(v92)];
          else
            v71 = &v91[(unsigned int)v92];
        }
        else
        {
          v56 = (unsigned __int64)v91;
          if ( v91 != v90 )
          {
            v59 = &v91[(unsigned int)v92];
            goto LABEL_79;
          }
          v59 = &v91[HIDWORD(v92)];
          v71 = v59;
        }
      }
      while ( v71 != v59 && (unsigned __int64)*v59 >= 0xFFFFFFFFFFFFFFFELL )
        ++v59;
LABEL_79:
      if ( v60 != v59 )
      {
        if ( v75 )
        {
          v61 = &v86[(unsigned int)v87 - 1];
          v62 = &v86[(unsigned int)(v55 + 1)];
          v63 = *v62;
          *v62 = *v61;
          *v61 = v63;
          LODWORD(v87) = v87 - 1;
        }
        v64 = 24;
        if ( (_DWORD)v55 != -2 )
          v64 = 24LL * (unsigned int)(2 * v55 + 3);
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v65 = *(_QWORD *)(a2 - 8);
        else
          v65 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        sub_157F2D0(*(_QWORD *)(v65 + v64), *(_QWORD *)(a2 + 40), 0);
        sub_15FFDB0(a2, a2, v55);
        goto LABEL_71;
      }
      if ( !v55 )
        goto LABEL_88;
    }
  }
  sub_1B44660(a4, v9);
  sub_157F2D0(v83->m128i_i64[1], *(_QWORD *)(a2 + 40), 0);
  v72 = a2;
  v23 = 1;
  sub_1B44FE0(v72);
  v22 = v83;
LABEL_21:
  if ( v22 )
    j_j___libc_free_0(v22, v85 - (_QWORD)v22);
  if ( base )
    j_j___libc_free_0(base, v82 - (_QWORD)base);
  return v23;
}
