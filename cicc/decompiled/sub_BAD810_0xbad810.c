// Function: sub_BAD810
// Address: 0xbad810
//
__m128i *__fastcall sub_BAD810(__m128i *a1, _QWORD *a2, __int64 m128i_i64, unsigned __int64 a4, __int64 a5)
{
  int v7; // esi
  __int64 v8; // r14
  char v9; // al
  char v10; // cl
  __int64 v11; // rsi
  __m128i *v12; // rdi
  size_t v13; // rdx
  __int64 v14; // r8
  char *v15; // rax
  __int64 v16; // rcx
  _QWORD *v17; // rcx
  __int64 v18; // r15
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rdi
  int v21; // eax
  __int64 v22; // r9
  _BYTE *v23; // rsi
  unsigned int v24; // ecx
  int v25; // edx
  unsigned __int64 v26; // r9
  int v27; // eax
  unsigned int v28; // r10d
  __int64 v29; // rax
  __int64 v30; // r11
  __int64 v31; // r9
  __int64 v32; // rcx
  __m128i *v33; // rax
  __m128i *v34; // rcx
  __m128i *v35; // rdx
  __int64 v36; // rcx
  __m128i *v37; // rax
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rcx
  __m128i *v41; // rax
  __m128i *v42; // rcx
  char v43; // al
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rcx
  __m128i *v48; // rax
  _QWORD *v49; // rsi
  __int64 v50; // rdx
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // rcx
  __m128i *v54; // rax
  __m128i *v55; // rcx
  __m128i *v56; // rdx
  __int64 v57; // rcx
  __m128i *v58; // rax
  __int64 v59; // rcx
  __m128i *v60; // rax
  size_t v61; // rcx
  unsigned int v62; // [rsp+0h] [rbp-190h]
  unsigned int v63; // [rsp+0h] [rbp-190h]
  unsigned int v64; // [rsp+0h] [rbp-190h]
  __m128i *v65; // [rsp+20h] [rbp-170h]
  unsigned __int64 v66; // [rsp+28h] [rbp-168h]
  __m128i v67; // [rsp+30h] [rbp-160h] BYREF
  __m128i *v68; // [rsp+40h] [rbp-150h] BYREF
  __int64 v69; // [rsp+48h] [rbp-148h]
  __m128i v70; // [rsp+50h] [rbp-140h] BYREF
  __m128i v71; // [rsp+60h] [rbp-130h] BYREF
  __int64 v72; // [rsp+70h] [rbp-120h] BYREF
  _QWORD *v73; // [rsp+80h] [rbp-110h] BYREF
  __int64 v74; // [rsp+88h] [rbp-108h]
  _QWORD v75[2]; // [rsp+90h] [rbp-100h] BYREF
  _QWORD v76[2]; // [rsp+A0h] [rbp-F0h] BYREF
  char v77[16]; // [rsp+B0h] [rbp-E0h] BYREF
  _QWORD v78[2]; // [rsp+C0h] [rbp-D0h] BYREF
  char v79[16]; // [rsp+D0h] [rbp-C0h] BYREF
  _QWORD *v80; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v81; // [rsp+E8h] [rbp-A8h]
  _QWORD v82[2]; // [rsp+F0h] [rbp-A0h] BYREF
  __m128i *v83; // [rsp+100h] [rbp-90h] BYREF
  __int64 v84; // [rsp+108h] [rbp-88h]
  __m128i v85; // [rsp+110h] [rbp-80h] BYREF
  char s[8]; // [rsp+120h] [rbp-70h] BYREF
  __int64 v87; // [rsp+128h] [rbp-68h]
  __m128i v88; // [rsp+130h] [rbp-60h] BYREF
  __m128i *v89; // [rsp+140h] [rbp-50h] BYREF
  size_t v90; // [rsp+148h] [rbp-48h]
  __m128i v91; // [rsp+150h] [rbp-40h] BYREF

  v7 = *(_DWORD *)(m128i_i64 + 8);
  if ( !v7 )
  {
    sub_BAC460(a1, a2, m128i_i64, a4, a5);
    return a1;
  }
  v8 = m128i_i64;
  if ( v7 != 1 )
  {
    v67.m128i_i8[0] = 0;
    v65 = &v67;
    v66 = 0;
    goto LABEL_52;
  }
  v9 = *(_BYTE *)(m128i_i64 + 60);
  v10 = *(_BYTE *)(m128i_i64 + 61);
  v89 = &v91;
  s[0] = (v9 & 1) + 48;
  s[1] = ((v9 & 2) != 0) + 48;
  s[2] = ((v9 & 4) != 0) + 48;
  s[3] = ((v9 & 8) != 0) + 48;
  s[4] = ((v9 & 0x10) != 0) + 48;
  s[5] = ((v9 & 0x20) != 0) + 48;
  s[6] = ((v9 & 0x40) != 0) + 48;
  s[7] = (v9 < 0) + 48;
  LOBYTE(v87) = (v10 & 1) + 48;
  *(_WORD *)((char *)&v87 + 1) = (unsigned __int8)(((v10 & 2) != 0) + 48);
  LODWORD(v11) = 1;
  v12 = &v91;
  v13 = strlen(s);
  v15 = s;
  if ( (unsigned int)v13 >= 8 )
  {
    LODWORD(v45) = 0;
    v14 = (unsigned int)v13 & 0xFFFFFFF8;
    do
    {
      v46 = (unsigned int)v45;
      v45 = (unsigned int)(v45 + 8);
      *(__int64 *)((char *)v91.m128i_i64 + v46) = *(_QWORD *)&s[v46];
    }
    while ( (unsigned int)v45 < (unsigned int)v14 );
    v12 = (__m128i *)((char *)&v91 + v45);
    v15 = &s[v45];
  }
  v16 = 0;
  if ( (v13 & 4) != 0 )
  {
    v12->m128i_i32[0] = *(_DWORD *)v15;
    v16 = 2;
  }
  if ( (v13 & 2) != 0 )
  {
    v14 = *(unsigned __int16 *)&v15[v16 * 2];
    v12->m128i_i16[v16++] = v14;
  }
  if ( (v13 & 1) != 0 )
    v12->m128i_i8[v16 * 2] = v15[v16 * 2];
  v17 = (_QWORD *)*(unsigned int *)(v8 + 56);
  v90 = v13;
  v91.m128i_i8[v13] = 0;
  if ( (unsigned int)v17 > 9 )
  {
    if ( (unsigned int)v17 <= 0x63 )
    {
      v63 = (unsigned int)v17;
      v80 = v82;
      sub_2240A50(&v80, 2, 0, v17, v14);
      v23 = v80;
      v24 = v63;
    }
    else
    {
      if ( (unsigned int)v17 <= 0x3E7 )
      {
        v11 = 3;
        v18 = (unsigned int)v17;
      }
      else
      {
        v14 = 0x346DC5D63886594BLL;
        v18 = (unsigned int)v17;
        v19 = (unsigned int)v17;
        if ( (unsigned int)v17 > 0x270F )
        {
          while ( 1 )
          {
            v20 = v19;
            v21 = v11;
            v11 = (unsigned int)(v11 + 4);
            v19 /= 0x2710u;
            if ( v20 <= 0x1869F )
              break;
            if ( (unsigned int)v19 <= 0x63 )
            {
              v62 = (unsigned int)v17;
              v17 = v82;
              v11 = (unsigned int)(v21 + 5);
              v80 = v82;
              goto LABEL_20;
            }
            if ( (unsigned int)v19 <= 0x3E7 )
            {
              v11 = (unsigned int)(v21 + 6);
              break;
            }
            if ( (unsigned int)v19 <= 0x270F )
            {
              v11 = (unsigned int)(v21 + 7);
              break;
            }
          }
        }
        else
        {
          v11 = 4;
        }
      }
      v62 = (unsigned int)v17;
      v80 = v82;
LABEL_20:
      sub_2240A50(&v80, v11, 0, v17, v14);
      v22 = v18;
      v23 = v80;
      v24 = v62;
      v25 = v81 - 1;
      while ( 1 )
      {
        v26 = (unsigned __int64)(1374389535 * v22) >> 37;
        v27 = v24 - 100 * v26;
        v28 = v24;
        v24 = v26;
        v29 = (unsigned int)(2 * v27);
        v30 = (unsigned int)(v29 + 1);
        LOBYTE(v29) = a00010203040506[v29];
        v23[v25] = a00010203040506[v30];
        v31 = (unsigned int)(v25 - 1);
        v25 -= 2;
        v23[v31] = v29;
        if ( v28 <= 0x270F )
          break;
        v22 = v24;
      }
      if ( v28 <= 0x3E7 )
        goto LABEL_24;
    }
    v32 = 2 * v24;
    v23[1] = a00010203040506[(unsigned int)(v32 + 1)];
    *v23 = a00010203040506[v32];
    goto LABEL_25;
  }
  v64 = (unsigned int)v17;
  v80 = v82;
  sub_2240A50(&v80, 1, 0, v17, v14);
  v23 = v80;
  v24 = v64;
LABEL_24:
  v32 = v24 + 48;
  *v23 = v32;
LABEL_25:
  strcpy(v79, "inst: ");
  v78[0] = v79;
  v78[1] = 6;
  if ( (unsigned __int64)(v81 + 6) <= 0xF || v80 == v82 || (unsigned __int64)(v81 + 6) > v82[0] )
  {
    v33 = (__m128i *)sub_2241490(v78, v80, v81, v32);
    v83 = &v85;
    v34 = (__m128i *)v33->m128i_i64[0];
    v35 = v33 + 1;
    if ( (__m128i *)v33->m128i_i64[0] != &v33[1] )
    {
LABEL_29:
      v83 = v34;
      v85.m128i_i64[0] = v33[1].m128i_i64[0];
      goto LABEL_30;
    }
  }
  else
  {
    v33 = (__m128i *)sub_2241130(&v80, 0, 0, v79, 6);
    v35 = v33 + 1;
    v83 = &v85;
    v34 = (__m128i *)v33->m128i_i64[0];
    if ( (__m128i *)v33->m128i_i64[0] != &v33[1] )
      goto LABEL_29;
  }
  v85 = _mm_loadu_si128(v33 + 1);
LABEL_30:
  v36 = v33->m128i_i64[1];
  v84 = v36;
  v33->m128i_i64[0] = (__int64)v35;
  v33->m128i_i64[1] = 0;
  v33[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v84) <= 6 )
    goto LABEL_125;
  v37 = (__m128i *)sub_2241490(&v83, ", ffl: ", 7, v36);
  *(_QWORD *)s = &v88;
  if ( (__m128i *)v37->m128i_i64[0] == &v37[1] )
  {
    v88 = _mm_loadu_si128(v37 + 1);
  }
  else
  {
    *(_QWORD *)s = v37->m128i_i64[0];
    v88.m128i_i64[0] = v37[1].m128i_i64[0];
  }
  v87 = v37->m128i_i64[1];
  v37->m128i_i64[0] = (__int64)v37[1].m128i_i64;
  v37->m128i_i64[1] = 0;
  v37[1].m128i_i8[0] = 0;
  v38 = 15;
  v39 = 15;
  if ( *(__m128i **)s != &v88 )
    v39 = v88.m128i_i64[0];
  v40 = v87 + v90;
  if ( v87 + v90 > v39 )
  {
    if ( v89 != &v91 )
      v38 = v91.m128i_i64[0];
    if ( v40 <= v38 )
    {
      v41 = (__m128i *)sub_2241130(&v89, 0, 0, *(_QWORD *)s, v87);
      v65 = &v67;
      v42 = (__m128i *)v41->m128i_i64[0];
      m128i_i64 = (__int64)v41[1].m128i_i64;
      if ( (__m128i *)v41->m128i_i64[0] != &v41[1] )
        goto LABEL_40;
LABEL_67:
      v67 = _mm_loadu_si128(v41 + 1);
      goto LABEL_41;
    }
  }
  v41 = (__m128i *)sub_2241490(s, v89, v90, v40);
  v65 = &v67;
  v42 = (__m128i *)v41->m128i_i64[0];
  m128i_i64 = (__int64)v41[1].m128i_i64;
  if ( (__m128i *)v41->m128i_i64[0] == &v41[1] )
    goto LABEL_67;
LABEL_40:
  v65 = v42;
  v67.m128i_i64[0] = v41[1].m128i_i64[0];
LABEL_41:
  a4 = v41->m128i_u64[1];
  v66 = a4;
  v41->m128i_i64[0] = m128i_i64;
  v41->m128i_i64[1] = 0;
  v41[1].m128i_i8[0] = 0;
  if ( *(__m128i **)s != &v88 )
    j_j___libc_free_0(*(_QWORD *)s, v88.m128i_i64[0] + 1);
  if ( v83 != &v85 )
    j_j___libc_free_0(v83, v85.m128i_i64[0] + 1);
  if ( (char *)v78[0] != v79 )
    j_j___libc_free_0(v78[0], *(_QWORD *)v79 + 1LL);
  if ( v80 != v82 )
    j_j___libc_free_0(v80, v82[0] + 1LL);
  if ( v89 != &v91 )
    j_j___libc_free_0(v89, v91.m128i_i64[0] + 1);
LABEL_52:
  v43 = *(_BYTE *)(v8 + 12) & 0xF;
  v73 = v75;
  switch ( v43 )
  {
    case 0:
      a5 = 28274;
      strcpy((char *)v75, "extern");
      v74 = 6;
      break;
    case 1:
      strcpy((char *)v75, "av_ext");
      v74 = 6;
      break;
    case 2:
      strcpy((char *)v75, "linkonce");
      v74 = 8;
      break;
    case 3:
      strcpy((char *)v75, "linkonce_odr");
      v74 = 12;
      break;
    case 4:
      strcpy((char *)v75, "weak");
      v74 = 4;
      break;
    case 5:
      strcpy((char *)v75, "weak_odr");
      v74 = 8;
      break;
    case 6:
      strcpy((char *)v75, "appending");
      v74 = 9;
      break;
    case 7:
      strcpy((char *)v75, "internal");
      v74 = 8;
      break;
    case 8:
      strcpy((char *)v75, "private");
      v74 = 7;
      break;
    case 9:
      a4 = 24933;
      strcpy((char *)v75, "extern_weak");
      v74 = 11;
      break;
    case 10:
      m128i_i64 = 28271;
      strcpy((char *)v75, "common");
      v74 = 6;
      break;
    default:
      v73 = v75;
      strcpy((char *)v75, "<unknown>");
      v74 = 9;
      break;
  }
  sub_BAC460(&v71, a2, m128i_i64, a4, a5);
  if ( v71.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_125;
  v48 = (__m128i *)sub_2241490(&v71, "|", 1, v47);
  v89 = &v91;
  if ( (__m128i *)v48->m128i_i64[0] == &v48[1] )
  {
    v91 = _mm_loadu_si128(v48 + 1);
  }
  else
  {
    v89 = (__m128i *)v48->m128i_i64[0];
    v91.m128i_i64[0] = v48[1].m128i_i64[0];
  }
  v90 = v48->m128i_u64[1];
  v48->m128i_i64[0] = (__int64)v48[1].m128i_i64;
  v49 = v73;
  v48->m128i_i64[1] = 0;
  v50 = v74;
  v48[1].m128i_i8[0] = 0;
  v51 = 15;
  v52 = 15;
  if ( v89 != &v91 )
    v52 = v91.m128i_i64[0];
  v53 = v90 + v50;
  if ( v90 + v50 <= v52 )
    goto LABEL_80;
  if ( v49 != v75 )
    v51 = v75[0];
  if ( v53 <= v51 )
  {
    v54 = (__m128i *)sub_2241130(&v73, 0, 0, v89, v90);
    v68 = &v70;
    v55 = (__m128i *)v54->m128i_i64[0];
    v56 = v54 + 1;
    if ( (__m128i *)v54->m128i_i64[0] != &v54[1] )
      goto LABEL_81;
  }
  else
  {
LABEL_80:
    v54 = (__m128i *)sub_2241490(&v89, v49, v50, v53);
    v68 = &v70;
    v55 = (__m128i *)v54->m128i_i64[0];
    v56 = v54 + 1;
    if ( (__m128i *)v54->m128i_i64[0] != &v54[1] )
    {
LABEL_81:
      v68 = v55;
      v70.m128i_i64[0] = v54[1].m128i_i64[0];
      goto LABEL_82;
    }
  }
  v70 = _mm_loadu_si128(v54 + 1);
LABEL_82:
  v57 = v54->m128i_i64[1];
  v69 = v57;
  v54->m128i_i64[0] = (__int64)v56;
  v54->m128i_i64[1] = 0;
  v54[1].m128i_i8[0] = 0;
  if ( v89 != &v91 )
    j_j___libc_free_0(v89, v91.m128i_i64[0] + 1);
  if ( (__int64 *)v71.m128i_i64[0] != &v72 )
    j_j___libc_free_0(v71.m128i_i64[0], v72 + 1);
  if ( v73 != v75 )
    j_j___libc_free_0(v73, v75[0] + 1LL);
  if ( v66 )
  {
    v76[0] = v77;
    strcpy(v77, " (");
    v76[1] = 2;
    v58 = (__m128i *)sub_2241490(v76, v65, v66, v57);
    *(_QWORD *)s = &v88;
    if ( (__m128i *)v58->m128i_i64[0] == &v58[1] )
    {
      v88 = _mm_loadu_si128(v58 + 1);
    }
    else
    {
      *(_QWORD *)s = v58->m128i_i64[0];
      v88.m128i_i64[0] = v58[1].m128i_i64[0];
    }
    v59 = v58->m128i_i64[1];
    v87 = v59;
    v58->m128i_i64[0] = (__int64)v58[1].m128i_i64;
    v58->m128i_i64[1] = 0;
    v58[1].m128i_i8[0] = 0;
    if ( v87 != 0x3FFFFFFFFFFFFFFFLL )
    {
      v60 = (__m128i *)sub_2241490(s, ")", 1, v59);
      v89 = &v91;
      if ( (__m128i *)v60->m128i_i64[0] == &v60[1] )
      {
        v91 = _mm_loadu_si128(v60 + 1);
      }
      else
      {
        v89 = (__m128i *)v60->m128i_i64[0];
        v91.m128i_i64[0] = v60[1].m128i_i64[0];
      }
      v90 = v60->m128i_u64[1];
      v61 = v90;
      v60->m128i_i64[0] = (__int64)v60[1].m128i_i64;
      v60->m128i_i64[1] = 0;
      v60[1].m128i_i8[0] = 0;
      sub_2241490(&v68, v89, v90, v61);
      if ( v89 != &v91 )
        j_j___libc_free_0(v89, v91.m128i_i64[0] + 1);
      if ( *(__m128i **)s != &v88 )
        j_j___libc_free_0(*(_QWORD *)s, v88.m128i_i64[0] + 1);
      if ( (char *)v76[0] != v77 )
        j_j___libc_free_0(v76[0], *(_QWORD *)v77 + 1LL);
      goto LABEL_89;
    }
LABEL_125:
    sub_4262D8((__int64)"basic_string::append");
  }
LABEL_89:
  if ( v69 == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_125;
  sub_2241490(&v68, "}", 1, v57);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( v68 == &v70 )
  {
    a1[1] = _mm_load_si128(&v70);
  }
  else
  {
    a1->m128i_i64[0] = (__int64)v68;
    a1[1].m128i_i64[0] = v70.m128i_i64[0];
  }
  a1->m128i_i64[1] = v69;
  if ( v65 != &v67 )
    j_j___libc_free_0(v65, v67.m128i_i64[0] + 1);
  return a1;
}
