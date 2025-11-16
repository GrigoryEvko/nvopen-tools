// Function: sub_27E3B20
// Address: 0x27e3b20
//
__int64 __fastcall sub_27E3B20(__int64 *a1, __int64 a2, __int64 **a3, __int64 a4, char *a5)
{
  __int64 *v6; // r14
  __int64 *v7; // r13
  unsigned int v8; // r12d
  unsigned int v9; // edi
  __int64 *v10; // rdx
  __int64 v11; // r8
  __int64 v12; // rbx
  unsigned int v13; // r12d
  unsigned __int64 v14; // rcx
  unsigned int v15; // esi
  int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r9
  int v19; // r8d
  __int64 *v20; // rdi
  const __m128i *v21; // rax
  size_t v22; // r12
  const __m128i *v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // rcx
  unsigned __int64 v32; // r10
  __int64 *v33; // r9
  const __m128i *v34; // rsi
  __int64 v35; // rbx
  __int64 v36; // rdx
  __int64 v37; // r13
  __int64 *v38; // r12
  __int64 v39; // r14
  unsigned __int64 v40; // r15
  __int64 v41; // rax
  __m128i *v42; // rsi
  __m128i *v43; // rdx
  __m128i *v44; // rsi
  __int64 *v45; // rdx
  __int64 v46; // r11
  bool v47; // cf
  __int64 v48; // r12
  int i; // edx
  int v51; // r9d
  int v52; // r11d
  __m128i *v53; // rax
  __m128i *v54; // rdi
  __int64 *v55; // rsi
  unsigned int v56; // r12d
  int v57; // edi
  __int64 v58; // r8
  unsigned __int64 v59; // [rsp+0h] [rbp-100h]
  unsigned __int64 v60; // [rsp+0h] [rbp-100h]
  __int64 *v62; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v64; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v65; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v66; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v68; // [rsp+28h] [rbp-D8h]
  __int64 v69; // [rsp+28h] [rbp-D8h]
  __int64 v70; // [rsp+28h] [rbp-D8h]
  __int64 *v71; // [rsp+28h] [rbp-D8h]
  char *sa; // [rsp+30h] [rbp-D0h]
  char *sb; // [rsp+30h] [rbp-D0h]
  char *sc; // [rsp+30h] [rbp-D0h]
  char *sd; // [rsp+30h] [rbp-D0h]
  __int64 *v77; // [rsp+38h] [rbp-C8h]
  __int64 v78; // [rsp+40h] [rbp-C0h]
  __int64 *v79; // [rsp+40h] [rbp-C0h]
  __int64 *v80; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v81; // [rsp+58h] [rbp-A8h]
  _BYTE v82[16]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v83; // [rsp+70h] [rbp-90h] BYREF
  __int64 v84; // [rsp+78h] [rbp-88h]
  __int64 v85; // [rsp+80h] [rbp-80h]
  unsigned int v86; // [rsp+88h] [rbp-78h]
  __m128i v87; // [rsp+90h] [rbp-70h] BYREF
  __m128i v88; // [rsp+A0h] [rbp-60h] BYREF
  __m128i *v89; // [rsp+B0h] [rbp-50h] BYREF
  __m128i *v90; // [rsp+B8h] [rbp-48h]
  _QWORD v91[8]; // [rsp+C0h] [rbp-40h] BYREF

  v80 = (__int64 *)v82;
  v81 = 0x200000000LL;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v6 = (__int64 *)sub_27DD5D0(a1);
  if ( v6 )
  {
    v78 = sub_27DE090((__int64)a1, 1);
    v77 = (__int64 *)&a3[a4];
    if ( v77 != (__int64 *)a3 )
    {
      v7 = (__int64 *)a3;
      while ( 1 )
      {
        v12 = *v7;
        v13 = sub_FF0430(v78, *v7, a2);
        v89 = (__m128i *)sub_FDD860(v6, v12);
        v14 = sub_1098D20((unsigned __int64 *)&v89, v13);
        if ( !v86 )
          break;
        v8 = ((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4);
        v9 = (v86 - 1) & v8;
        v10 = (__int64 *)(v84 + 16LL * v9);
        v11 = *v10;
        if ( v12 != *v10 )
        {
          v52 = 1;
          v17 = 0;
          while ( v11 != -4096 )
          {
            if ( !v17 && v11 == -8192 )
              v17 = v10;
            v9 = (v86 - 1) & (v52 + v9);
            v10 = (__int64 *)(v84 + 16LL * v9);
            v11 = *v10;
            if ( v12 == *v10 )
              goto LABEL_5;
            ++v52;
          }
          if ( !v17 )
            v17 = v10;
          ++v83;
          v16 = v85 + 1;
          if ( 4 * ((int)v85 + 1) < 3 * v86 )
          {
            if ( v86 - HIDWORD(v85) - v16 <= v86 >> 3 )
            {
              v60 = v14;
              sub_27E3940((__int64)&v83, v86);
              if ( !v86 )
              {
LABEL_110:
                LODWORD(v85) = v85 + 1;
                BUG();
              }
              v55 = 0;
              v56 = (v86 - 1) & v8;
              v14 = v60;
              v16 = v85 + 1;
              v57 = 1;
              v17 = (__int64 *)(v84 + 16LL * v56);
              v58 = *v17;
              if ( v12 != *v17 )
              {
                while ( v58 != -4096 )
                {
                  if ( !v55 && v58 == -8192 )
                    v55 = v17;
                  v56 = (v86 - 1) & (v57 + v56);
                  v17 = (__int64 *)(v84 + 16LL * v56);
                  v58 = *v17;
                  if ( v12 == *v17 )
                    goto LABEL_70;
                  ++v57;
                }
                if ( v55 )
                  v17 = v55;
              }
            }
            goto LABEL_70;
          }
LABEL_8:
          v59 = v14;
          sub_27E3940((__int64)&v83, 2 * v86);
          if ( !v86 )
            goto LABEL_110;
          v14 = v59;
          v15 = (v86 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v16 = v85 + 1;
          v17 = (__int64 *)(v84 + 16LL * v15);
          v18 = *v17;
          if ( v12 != *v17 )
          {
            v19 = 1;
            v20 = 0;
            while ( v18 != -4096 )
            {
              if ( v18 == -8192 && !v20 )
                v20 = v17;
              v15 = (v86 - 1) & (v19 + v15);
              v17 = (__int64 *)(v84 + 16LL * v15);
              v18 = *v17;
              if ( v12 == *v17 )
                goto LABEL_70;
              ++v19;
            }
            if ( v20 )
              v17 = v20;
          }
LABEL_70:
          LODWORD(v85) = v16;
          if ( *v17 != -4096 )
            --HIDWORD(v85);
          *v17 = v12;
          v17[1] = v14;
        }
LABEL_5:
        if ( v77 == ++v7 )
          goto LABEL_15;
      }
      ++v83;
      goto LABEL_8;
    }
  }
LABEL_15:
  if ( sub_AA5E90(a2) )
  {
    v89 = (__m128i *)v91;
    if ( !a5 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v21 = (const __m128i *)strlen(a5);
    v87.m128i_i64[0] = (__int64)v21;
    v22 = (size_t)v21;
    if ( (unsigned __int64)v21 > 0xF )
    {
      v89 = (__m128i *)sub_22409D0((__int64)&v89, (unsigned __int64 *)&v87, 0);
      v54 = v89;
      v91[0] = v87.m128i_i64[0];
    }
    else
    {
      if ( v21 == (const __m128i *)1 )
      {
        LOBYTE(v91[0]) = *a5;
        v23 = (const __m128i *)v91;
LABEL_75:
        v90 = (__m128i *)v21;
        v21->m128i_i8[(_QWORD)v23] = 0;
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - (_QWORD)v90) <= 8 )
          sub_4262D8((__int64)"basic_string::append");
        v53 = (__m128i *)sub_2241490((unsigned __int64 *)&v89, ".split-lp", 9u);
        v87.m128i_i64[0] = (__int64)&v88;
        if ( (__m128i *)v53->m128i_i64[0] == &v53[1] )
        {
          v88 = _mm_loadu_si128(v53 + 1);
        }
        else
        {
          v87.m128i_i64[0] = v53->m128i_i64[0];
          v88.m128i_i64[0] = v53[1].m128i_i64[0];
        }
        v87.m128i_i64[1] = v53->m128i_i64[1];
        v53->m128i_i64[0] = (__int64)v53[1].m128i_i64;
        v53->m128i_i64[1] = 0;
        v53[1].m128i_i8[0] = 0;
        if ( v89 != (__m128i *)v91 )
          j_j___libc_free_0((unsigned __int64)v89);
        sub_F40790(a2, a3, a4, a5, v87.m128i_i64[0], (__int64)&v80, 0, 0, 0, 0);
        if ( (__m128i *)v87.m128i_i64[0] != &v88 )
          j_j___libc_free_0(v87.m128i_u64[0]);
        v29 = (unsigned int)v81;
        goto LABEL_23;
      }
      if ( !v21 )
      {
        v23 = (const __m128i *)v91;
        goto LABEL_75;
      }
      v54 = (__m128i *)v91;
    }
    memcpy(v54, a5, v22);
    v21 = (const __m128i *)v87.m128i_i64[0];
    v23 = v89;
    goto LABEL_75;
  }
  v24 = sub_F41DE0(a2, a3, a4, a5, 0, 0, 0, 0);
  v27 = (unsigned int)v81;
  v28 = (unsigned int)v81 + 1LL;
  if ( v28 > HIDWORD(v81) )
  {
    sub_C8D5F0((__int64)&v80, v82, v28, 8u, v25, v26);
    v27 = (unsigned int)v81;
  }
  v80[v27] = v24;
  v29 = (unsigned int)(v81 + 1);
  LODWORD(v81) = v81 + 1;
LABEL_23:
  v89 = 0;
  v90 = 0;
  v91[0] = 0;
  sub_F58D10((const __m128i **)&v89, v29 + 2 * a4);
  v31 = (__int64)&v87;
  v32 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v33 = v6;
  v79 = v80;
  v62 = &v80[(unsigned int)v81];
  if ( v62 != v80 )
  {
    while ( 1 )
    {
      v34 = v90;
      v30 = *v79;
      v87.m128i_i64[1] = v32;
      v87.m128i_i64[0] = v30;
      if ( v90 == (__m128i *)v91[0] )
      {
        v66 = v32;
        v71 = v33;
        sd = (char *)v30;
        sub_F38BA0((const __m128i **)&v89, v90, &v87);
        v32 = v66;
        v33 = v71;
        v30 = (__int64)sd;
      }
      else
      {
        if ( v90 )
        {
          *v90 = _mm_load_si128(&v87);
          v34 = v90;
        }
        v90 = (__m128i *)&v34[1];
      }
      v35 = *(_QWORD *)(v30 + 16);
      if ( v35 )
        break;
LABEL_54:
      v37 = 0;
LABEL_45:
      if ( v33 )
      {
        v68 = v32;
        sa = (char *)v33;
        sub_FE1040(v33, v30, v37);
        v32 = v68;
        v33 = (__int64 *)sa;
      }
      if ( v62 == ++v79 )
        goto LABEL_48;
    }
    while ( 1 )
    {
      v36 = *(_QWORD *)(v35 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v36 - 30) <= 0xAu )
        break;
      v35 = *(_QWORD *)(v35 + 8);
      if ( !v35 )
        goto LABEL_54;
    }
    v37 = 0;
    v38 = v33;
    v39 = v32 | 4;
    v40 = v30 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_32:
    v41 = *(_QWORD *)(v36 + 40);
    v42 = v90;
    v87.m128i_i64[1] = v39;
    v43 = (__m128i *)v91[0];
    v87.m128i_i64[0] = v41;
    if ( v90 == (__m128i *)v91[0] )
    {
      v64 = v32;
      v69 = v30;
      sb = (char *)v41;
      sub_F38BA0((const __m128i **)&v89, v90, &v87);
      v41 = (__int64)sb;
      v44 = v90;
      v87.m128i_i64[1] = v40;
      v30 = v69;
      v87.m128i_i64[0] = (__int64)sb;
      v32 = v64;
      if ( (__m128i *)v91[0] != v90 )
      {
        if ( !v90 )
          goto LABEL_37;
        goto LABEL_36;
      }
    }
    else
    {
      if ( v90 )
      {
        *v90 = _mm_load_si128(&v87);
        v43 = (__m128i *)v91[0];
        v42 = v90;
      }
      v44 = v42 + 1;
      v87.m128i_i64[0] = v41;
      v90 = v44;
      v87.m128i_i64[1] = v40;
      if ( v43 != v44 )
      {
LABEL_36:
        *v44 = _mm_load_si128(&v87);
        v44 = v90;
LABEL_37:
        v90 = v44 + 1;
        goto LABEL_38;
      }
    }
    v65 = v32;
    v70 = v30;
    sc = (char *)v41;
    sub_F38BA0((const __m128i **)&v89, v44, &v87);
    v32 = v65;
    v30 = v70;
    v41 = (__int64)sc;
LABEL_38:
    if ( v38 && v86 )
    {
      v31 = (v86 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v45 = (__int64 *)(v84 + 16 * v31);
      v46 = *v45;
      if ( v41 != *v45 )
      {
        for ( i = 1; ; i = v51 )
        {
          if ( v46 == -4096 )
            goto LABEL_43;
          v51 = i + 1;
          v31 = (v86 - 1) & (i + (_DWORD)v31);
          v45 = (__int64 *)(v84 + 16LL * (unsigned int)v31);
          v46 = *v45;
          if ( v41 == *v45 )
            break;
        }
      }
      v47 = __CFADD__(v45[1], v37);
      v37 += v45[1];
      if ( v47 )
        v37 = -1;
    }
LABEL_43:
    while ( 1 )
    {
      v35 = *(_QWORD *)(v35 + 8);
      if ( !v35 )
        break;
      v36 = *(_QWORD *)(v35 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v36 - 30) <= 0xAu )
        goto LABEL_32;
    }
    v33 = v38;
    goto LABEL_45;
  }
LABEL_48:
  sub_FFDB80(a1[6], (unsigned __int64 *)v89, v90 - v89, v31, v30, (__int64)v33);
  v48 = *v80;
  if ( v89 )
    j_j___libc_free_0((unsigned __int64)v89);
  sub_C7D6A0(v84, 16LL * v86, 8);
  if ( v80 != (__int64 *)v82 )
    _libc_free((unsigned __int64)v80);
  return v48;
}
