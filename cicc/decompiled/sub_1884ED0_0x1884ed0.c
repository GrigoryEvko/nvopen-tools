// Function: sub_1884ED0
// Address: 0x1884ed0
//
__int64 __fastcall sub_1884ED0(char *a1, char *a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  _QWORD *v4; // r14
  char *v5; // rbx
  char *v6; // r15
  _QWORD *v7; // r12
  char *v8; // r8
  __int64 v9; // rdx
  signed __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rcx
  bool v13; // cf
  unsigned __int64 v14; // rax
  __int64 v15; // r13
  char *v16; // rcx
  char *v17; // r13
  unsigned __int64 v18; // rdx
  size_t v19; // r14
  char v20; // al
  char *v21; // rdi
  char *v22; // rcx
  __int64 v23; // rax
  char *v24; // r15
  __int64 v25; // r14
  size_t v26; // r14
  char *v27; // rcx
  char *v28; // r15
  __int64 v29; // r14
  size_t v30; // r14
  char *v31; // r15
  __int64 v32; // r14
  size_t v33; // r14
  __int64 v34; // rax
  const __m128i *v35; // r14
  const __m128i *v36; // rax
  __m128i *v37; // r15
  const __m128i *v38; // rbx
  const __m128i *v39; // r14
  char *v40; // rdi
  __int64 v41; // r12
  unsigned __int64 v42; // r12
  __m128i *v43; // r12
  __int64 v44; // r14
  __int64 v45; // rdi
  unsigned __int64 v46; // rcx
  char *v47; // r8
  __int64 v48; // rax
  const __m128i *v49; // r12
  const __m128i *v50; // r15
  __m128i *v51; // r14
  unsigned __int64 v52; // rbx
  char *v53; // rcx
  __int64 v54; // rbx
  char *v55; // rax
  __m128i *v56; // r12
  __int64 v57; // rdi
  unsigned __int64 v58; // rax
  __int64 v59; // r13
  __int64 v60; // rax
  __int64 v61; // [rsp+8h] [rbp-168h]
  char *v62; // [rsp+10h] [rbp-160h]
  char *v63; // [rsp+18h] [rbp-158h]
  char *v64; // [rsp+20h] [rbp-150h]
  signed __int64 v65; // [rsp+28h] [rbp-148h]
  const __m128i *v66; // [rsp+30h] [rbp-140h]
  __int64 v67; // [rsp+30h] [rbp-140h]
  char *v68; // [rsp+30h] [rbp-140h]
  char *v69; // [rsp+30h] [rbp-140h]
  __int64 v70; // [rsp+30h] [rbp-140h]
  __int64 v71; // [rsp+38h] [rbp-138h]
  char *v72; // [rsp+38h] [rbp-138h]
  char *v73; // [rsp+38h] [rbp-138h]
  char *v74; // [rsp+38h] [rbp-138h]
  signed __int64 v75; // [rsp+38h] [rbp-138h]
  char *v76; // [rsp+38h] [rbp-138h]
  char *v77; // [rsp+38h] [rbp-138h]
  __int64 *v78; // [rsp+40h] [rbp-130h]
  __int64 *v79; // [rsp+48h] [rbp-128h]
  char v80; // [rsp+5Fh] [rbp-111h] BYREF
  char *v81; // [rsp+60h] [rbp-110h] BYREF
  char v82; // [rsp+74h] [rbp-FCh] BYREF
  _BYTE v83[11]; // [rsp+75h] [rbp-FBh] BYREF
  char *v84; // [rsp+80h] [rbp-F0h] BYREF
  char *v85; // [rsp+88h] [rbp-E8h]
  char *v86; // [rsp+90h] [rbp-E0h]
  char *v87; // [rsp+A0h] [rbp-D0h] BYREF
  char *v88; // [rsp+A8h] [rbp-C8h]
  char *v89; // [rsp+B0h] [rbp-C0h] BYREF
  char *v90; // [rsp+B8h] [rbp-B8h]
  char *v91; // [rsp+C0h] [rbp-B0h]
  char *v92; // [rsp+C8h] [rbp-A8h]
  char *v93; // [rsp+D0h] [rbp-A0h]
  char *v94; // [rsp+D8h] [rbp-98h]
  char *v95; // [rsp+E0h] [rbp-90h]
  char *v96; // [rsp+E8h] [rbp-88h]
  char *v97; // [rsp+F0h] [rbp-80h]
  char *v98; // [rsp+F8h] [rbp-78h]
  char *v99; // [rsp+100h] [rbp-70h]
  __int64 v100; // [rsp+108h] [rbp-68h]
  __m128i *v101; // [rsp+110h] [rbp-60h]
  __int64 v102; // [rsp+118h] [rbp-58h]
  __int64 v103; // [rsp+120h] [rbp-50h]
  __m128i *v104; // [rsp+128h] [rbp-48h]
  __int64 v105; // [rsp+130h] [rbp-40h]

  result = *((_QWORD *)a2 + 3);
  v61 = (__int64)a1;
  v62 = a2 + 8;
  v63 = (char *)result;
  if ( (char *)result == a2 + 8 )
    return result;
  do
  {
    v84 = 0;
    v85 = 0;
    v86 = 0;
    v79 = (__int64 *)*((_QWORD *)v63 + 7);
    v78 = (__int64 *)*((_QWORD *)v63 + 8);
    if ( v79 == v78 )
      goto LABEL_80;
    do
    {
      v3 = *v79;
      if ( *(_DWORD *)(*v79 + 8) != 1 )
        goto LABEL_72;
      if ( *(_QWORD *)(v3 + 40) != *(_QWORD *)(v3 + 48) )
      {
        v71 = *v79;
        v4 = *(_QWORD **)(v3 + 40);
        v5 = 0;
        v6 = 0;
        v7 = *(_QWORD **)(v3 + 48);
        v8 = 0;
        while ( 1 )
        {
          while ( 1 )
          {
            v9 = *(_QWORD *)(*v4 & 0xFFFFFFFFFFFFFFF8LL);
            if ( v5 == v6 )
              break;
            if ( v6 )
              *(_QWORD *)v6 = v9;
            ++v4;
            v6 += 8;
            if ( v7 == v4 )
            {
LABEL_21:
              v3 = v71;
              v17 = v8;
              v18 = v6 - v8;
              v19 = v6 - v8;
              goto LABEL_22;
            }
          }
          v10 = v5 - v8;
          v11 = (v5 - v8) >> 3;
          if ( v11 == 0xFFFFFFFFFFFFFFFLL )
            sub_4262D8((__int64)"vector::_M_realloc_insert");
          v12 = 1;
          if ( v11 )
            v12 = (v5 - v8) >> 3;
          v13 = __CFADD__(v12, v11);
          v14 = v12 + v11;
          if ( v13 )
          {
            v59 = 0x7FFFFFFFFFFFFFF8LL;
          }
          else
          {
            if ( !v14 )
            {
              v15 = 0;
              v16 = 0;
              goto LABEL_16;
            }
            if ( v14 > 0xFFFFFFFFFFFFFFFLL )
              v14 = 0xFFFFFFFFFFFFFFFLL;
            v59 = 8 * v14;
          }
          a1 = (char *)v59;
          v64 = v8;
          v65 = v5 - v8;
          v70 = *(_QWORD *)(*v4 & 0xFFFFFFFFFFFFFFF8LL);
          v60 = sub_22077B0(v59);
          v9 = v70;
          v10 = v65;
          v8 = v64;
          v16 = (char *)v60;
          v15 = v60 + v59;
LABEL_16:
          if ( &v16[v10] )
            *(_QWORD *)&v16[v10] = v9;
          v6 = &v16[v10 + 8];
          if ( v10 > 0 )
          {
            v68 = v8;
            v55 = (char *)memmove(v16, v8, v10);
            v8 = v68;
            v16 = v55;
LABEL_98:
            a1 = v8;
            v69 = v16;
            a2 = (char *)(v5 - v8);
            j_j___libc_free_0(v8, v5 - v8);
            v16 = v69;
            goto LABEL_20;
          }
          if ( v8 )
            goto LABEL_98;
LABEL_20:
          ++v4;
          v5 = (char *)v15;
          v8 = v16;
          if ( v7 == v4 )
            goto LABEL_21;
        }
      }
      v19 = 0;
      v18 = 0;
      v5 = 0;
      v6 = 0;
      v17 = 0;
LABEL_22:
      LODWORD(v87) = *(_BYTE *)(v3 + 12) & 0xF;
      BYTE4(v87) = (*(_BYTE *)(v3 + 12) & 0x10) != 0;
      BYTE5(v87) = (*(_BYTE *)(v3 + 12) & 0x20) != 0;
      v20 = *(_BYTE *)(v3 + 12);
      v88 = 0;
      v89 = 0;
      v90 = 0;
      BYTE6(v87) = (v20 & 0x40) != 0;
      if ( v18 )
      {
        if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_123:
          sub_4261EA(a1, a2, v18);
        v21 = (char *)sub_22077B0(v19);
      }
      else
      {
        v21 = 0;
      }
      v22 = &v21[v19];
      v88 = v21;
      v90 = &v21[v19];
      if ( v6 != v17 )
      {
        a2 = v17;
        memcpy(v21, v17, v19);
        v22 = &v21[v19];
      }
      v89 = v22;
      v23 = *(_QWORD *)(v3 + 96);
      if ( !v23 )
        goto LABEL_104;
      v24 = *(char **)v23;
      v25 = *(_QWORD *)(v23 + 8);
      v91 = 0;
      v92 = 0;
      v93 = 0;
      v26 = v25 - (_QWORD)v24;
      if ( v26 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_122;
      if ( v26 )
      {
        a2 = v24;
        v91 = (char *)sub_22077B0(v26);
        v93 = &v91[v26];
        v72 = &v91[v26];
        memcpy(v91, v24, v26);
        v23 = *(_QWORD *)(v3 + 96);
        v27 = v72;
      }
      else
      {
LABEL_104:
        v91 = 0;
        v27 = 0;
        v93 = 0;
      }
      v92 = v27;
      if ( !v23 )
        goto LABEL_100;
      v28 = *(char **)(v23 + 24);
      v29 = *(_QWORD *)(v23 + 32);
      v94 = 0;
      v95 = 0;
      v96 = 0;
      v30 = v29 - (_QWORD)v28;
      if ( v30 > 0x7FFFFFFFFFFFFFF0LL )
        goto LABEL_122;
      if ( v30 )
      {
        a2 = v28;
        v94 = (char *)sub_22077B0(v30);
        v96 = &v94[v30];
        v73 = &v94[v30];
        memcpy(v94, v28, v30);
        v23 = *(_QWORD *)(v3 + 96);
        v95 = v73;
        if ( !v23 )
          goto LABEL_101;
      }
      else
      {
LABEL_100:
        v94 = 0;
        v96 = 0;
        v95 = 0;
        if ( !v23 )
          goto LABEL_101;
      }
      v31 = *(char **)(v23 + 48);
      v32 = *(_QWORD *)(v23 + 56);
      v97 = 0;
      v98 = 0;
      v33 = v32 - (_QWORD)v31;
      v99 = 0;
      if ( v33 > 0x7FFFFFFFFFFFFFF0LL )
        goto LABEL_122;
      if ( v33 )
      {
        a2 = v31;
        v97 = (char *)sub_22077B0(v33);
        v99 = &v97[v33];
        v74 = &v97[v33];
        memcpy(v97, v31, v33);
        v98 = v74;
        v34 = *(_QWORD *)(v3 + 96);
        if ( !v34 )
          goto LABEL_102;
        goto LABEL_38;
      }
LABEL_101:
      v97 = 0;
      v99 = 0;
      v98 = 0;
      v34 = *(_QWORD *)(v3 + 96);
      if ( !v34 )
        goto LABEL_102;
LABEL_38:
      v35 = *(const __m128i **)(v34 + 72);
      v36 = *(const __m128i **)(v34 + 80);
      v100 = 0;
      v101 = 0;
      v66 = v36;
      v102 = 0;
      if ( (unsigned __int64)((char *)v36 - (char *)v35) > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_122;
      if ( v36 == v35 )
      {
LABEL_102:
        v100 = 0;
        v102 = 0;
        v101 = 0;
        v48 = *(_QWORD *)(v3 + 96);
        if ( !v48 )
          goto LABEL_103;
        goto LABEL_84;
      }
      a1 = (char *)((char *)v36 - (char *)v35);
      v75 = (char *)v36 - (char *)v35;
      v100 = sub_22077B0((char *)v36 - (char *)v35);
      v37 = (__m128i *)v100;
      v102 = v100 + v75;
      if ( v35 != v66 )
      {
        v76 = v5;
        v38 = v35;
        v39 = v66;
        v67 = v3;
        do
        {
          if ( v37 )
          {
            *v37 = _mm_loadu_si128(v38);
            v18 = v38[1].m128i_i64[1] - v38[1].m128i_i64[0];
            v42 = v18;
            v37[1].m128i_i64[0] = 0;
            v37[1].m128i_i64[1] = 0;
            v37[2].m128i_i64[0] = 0;
            if ( v18 )
            {
              if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
                goto LABEL_123;
              v40 = (char *)sub_22077B0(v18);
            }
            else
            {
              v40 = 0;
            }
            v37[1].m128i_i64[0] = (__int64)v40;
            v37[1].m128i_i64[1] = (__int64)v40;
            v37[2].m128i_i64[0] = (__int64)&v40[v42];
            a2 = (char *)v38[1].m128i_i64[0];
            v41 = v38[1].m128i_i64[1] - (_QWORD)a2;
            if ( (char *)v38[1].m128i_i64[1] != a2 )
              v40 = (char *)memmove(v40, a2, v38[1].m128i_i64[1] - (_QWORD)a2);
            a1 = &v40[v41];
            v37[1].m128i_i64[1] = (__int64)a1;
          }
          v38 = (const __m128i *)((char *)v38 + 40);
          v37 = (__m128i *)((char *)v37 + 40);
        }
        while ( v39 != v38 );
        v5 = v76;
        v3 = v67;
      }
      v101 = v37;
      v48 = *(_QWORD *)(v3 + 96);
      if ( !v48 )
        goto LABEL_103;
LABEL_84:
      v49 = *(const __m128i **)(v48 + 104);
      v50 = *(const __m128i **)(v48 + 96);
      v103 = 0;
      v104 = 0;
      v105 = 0;
      if ( (unsigned __int64)((char *)v49 - (char *)v50) > 0x7FFFFFFFFFFFFFF8LL )
LABEL_122:
        sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
      if ( v49 == v50 )
      {
LABEL_103:
        v103 = 0;
        v51 = 0;
        v105 = 0;
        goto LABEL_52;
      }
      a1 = (char *)((char *)v49 - (char *)v50);
      v103 = sub_22077B0((char *)v49 - (char *)v50);
      v51 = (__m128i *)v103;
      v105 = v103 + (char *)v49 - (char *)v50;
      if ( v50 != v49 )
      {
        v77 = v5;
        do
        {
          if ( v51 )
          {
            *v51 = _mm_loadu_si128(v50);
            v18 = v50[1].m128i_i64[1] - v50[1].m128i_i64[0];
            v51[1].m128i_i64[0] = 0;
            v51[1].m128i_i64[1] = 0;
            v51[2].m128i_i64[0] = 0;
            if ( v18 )
            {
              v52 = v18;
              if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
                goto LABEL_123;
              a1 = (char *)v18;
              v53 = (char *)sub_22077B0(v18);
            }
            else
            {
              v52 = 0;
              v53 = 0;
            }
            v51[1].m128i_i64[0] = (__int64)v53;
            v51[1].m128i_i64[1] = (__int64)v53;
            v51[2].m128i_i64[0] = (__int64)&v53[v52];
            a2 = (char *)v50[1].m128i_i64[0];
            v54 = v50[1].m128i_i64[1] - (_QWORD)a2;
            if ( (char *)v50[1].m128i_i64[1] != a2 )
            {
              a1 = v53;
              v53 = (char *)memmove(v53, a2, v50[1].m128i_i64[1] - (_QWORD)a2);
            }
            v51[1].m128i_i64[1] = (__int64)&v53[v54];
          }
          v50 = (const __m128i *)((char *)v50 + 40);
          v51 = (__m128i *)((char *)v51 + 40);
        }
        while ( v49 != v50 );
        v5 = v77;
      }
LABEL_52:
      v104 = v51;
      a2 = v85;
      if ( v85 == v86 )
      {
        sub_1880140((__int64 *)&v84, v85, (int *)&v87);
        v51 = v104;
        v56 = (__m128i *)v103;
        goto LABEL_106;
      }
      if ( !v85 )
      {
        v85 = (char *)152;
        v56 = (__m128i *)v103;
LABEL_106:
        if ( v51 != v56 )
        {
          do
          {
            v57 = v56[1].m128i_i64[0];
            if ( v57 )
            {
              a2 = (char *)(v56[2].m128i_i64[0] - v57);
              j_j___libc_free_0(v57, a2);
            }
            v56 = (__m128i *)((char *)v56 + 40);
          }
          while ( v51 != v56 );
          v56 = (__m128i *)v103;
        }
        if ( v56 )
        {
          a2 = (char *)(v105 - (_QWORD)v56);
          j_j___libc_free_0(v56, v105 - (_QWORD)v56);
        }
        goto LABEL_55;
      }
      *(_DWORD *)v85 = (_DWORD)v87;
      *((_WORD *)a2 + 2) = WORD2(v87);
      a2[6] = BYTE6(v87);
      *((_QWORD *)a2 + 1) = v88;
      *((_QWORD *)a2 + 2) = v89;
      *((_QWORD *)a2 + 3) = v90;
      v90 = 0;
      v89 = 0;
      v88 = 0;
      *((_QWORD *)a2 + 4) = v91;
      *((_QWORD *)a2 + 5) = v92;
      *((_QWORD *)a2 + 6) = v93;
      v93 = 0;
      v92 = 0;
      v91 = 0;
      *((_QWORD *)a2 + 7) = v94;
      *((_QWORD *)a2 + 8) = v95;
      *((_QWORD *)a2 + 9) = v96;
      v96 = 0;
      v95 = 0;
      v94 = 0;
      *((_QWORD *)a2 + 10) = v97;
      v85 += 152;
      *((_QWORD *)a2 + 11) = v98;
      *((_QWORD *)a2 + 12) = v99;
      v99 = 0;
      v98 = 0;
      v97 = 0;
      *((_QWORD *)a2 + 13) = v100;
      *((_QWORD *)a2 + 14) = v101;
      *((_QWORD *)a2 + 15) = v102;
      v102 = 0;
      v101 = 0;
      v100 = 0;
      *((_QWORD *)a2 + 16) = v103;
      *((_QWORD *)a2 + 17) = v104;
      *((_QWORD *)a2 + 18) = v105;
LABEL_55:
      v43 = v101;
      v44 = v100;
      if ( v101 != (__m128i *)v100 )
      {
        do
        {
          v45 = *(_QWORD *)(v44 + 16);
          if ( v45 )
          {
            a2 = (char *)(*(_QWORD *)(v44 + 32) - v45);
            j_j___libc_free_0(v45, a2);
          }
          v44 += 40;
        }
        while ( v43 != (__m128i *)v44 );
        v44 = v100;
      }
      if ( v44 )
      {
        a2 = (char *)(v102 - v44);
        j_j___libc_free_0(v44, v102 - v44);
      }
      if ( v97 )
      {
        a2 = (char *)(v99 - v97);
        j_j___libc_free_0(v97, v99 - v97);
      }
      if ( v94 )
      {
        a2 = (char *)(v96 - v94);
        j_j___libc_free_0(v94, v96 - v94);
      }
      if ( v91 )
      {
        a2 = (char *)(v93 - v91);
        j_j___libc_free_0(v91, v93 - v91);
      }
      a1 = v88;
      if ( v88 )
      {
        a2 = (char *)(v90 - v88);
        j_j___libc_free_0(v88, v90 - v88);
      }
      if ( v17 )
      {
        a1 = v17;
        a2 = (char *)(v5 - v17);
        j_j___libc_free_0(v17, v5 - v17);
      }
LABEL_72:
      ++v79;
    }
    while ( v78 != v79 );
    if ( v85 != v84 )
    {
      v46 = *((_QWORD *)v63 + 4);
      if ( v46 )
      {
        v47 = v83;
        do
        {
          *--v47 = v46 % 0xA + 48;
          v58 = v46;
          v46 /= 0xAu;
        }
        while ( v58 > 9 );
      }
      else
      {
        v82 = 48;
        v47 = &v82;
      }
      v87 = (char *)&v89;
      sub_1872C70((__int64 *)&v87, v47, (__int64)v83);
      a2 = v87;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, char **))(*(_QWORD *)v61 + 120LL))(
             v61,
             v87,
             1,
             0,
             &v80,
             &v81) )
      {
        sub_18843B0(v61, (__int64 *)&v84);
        a2 = v81;
        (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)v61 + 128LL))(v61, v81);
      }
      if ( v87 != (char *)&v89 )
      {
        a2 = v89 + 1;
        j_j___libc_free_0(v87, v89 + 1);
      }
    }
LABEL_80:
    sub_187D570(&v84);
    a1 = v63;
    result = sub_220EEE0(v63);
    v63 = (char *)result;
  }
  while ( v62 != (char *)result );
  return result;
}
