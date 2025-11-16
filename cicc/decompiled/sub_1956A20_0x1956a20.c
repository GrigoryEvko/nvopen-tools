// Function: sub_1956A20
// Address: 0x1956a20
//
__int64 __fastcall sub_1956A20(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, const char *a5)
{
  __int64 v5; // r15
  __int64 v6; // r14
  bool v7; // zf
  __int64 *v9; // r12
  unsigned int v10; // edi
  __int64 *v11; // rax
  __int64 v12; // r8
  __int64 v13; // rbx
  int v14; // r13d
  __int64 v15; // r13
  __int64 v16; // rsi
  int v17; // eax
  __int64 *v18; // rdx
  __int64 v19; // r9
  int v20; // r8d
  __int64 *v21; // rdi
  const __m128i *v22; // rax
  size_t v23; // r12
  const __m128i *v24; // rdx
  __m128i *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rbx
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // rax
  const __m128i *v31; // rsi
  __int64 v32; // rax
  __int64 v33; // r15
  _QWORD *v34; // rax
  __int64 v35; // r13
  __int64 v36; // r15
  __m128i *v37; // rsi
  __m128i *v38; // rdx
  __m128i *v39; // rsi
  __int64 v40; // r8
  unsigned int v41; // edi
  __int64 *v42; // rdx
  __int64 v43; // r10
  __int64 v44; // r12
  int v46; // edx
  int v47; // ecx
  __m128i *v48; // rdi
  int v49; // eax
  __int64 *v50; // rsi
  __int64 v51; // r14
  int v52; // edi
  __int64 v53; // r8
  __int64 *v55; // [rsp+10h] [rbp-F0h]
  unsigned __int64 sa; // [rsp+18h] [rbp-E8h]
  __int64 *v59; // [rsp+30h] [rbp-D0h]
  __int64 v60; // [rsp+30h] [rbp-D0h]
  __int64 *v62; // [rsp+38h] [rbp-C8h]
  __int64 v63; // [rsp+48h] [rbp-B8h] BYREF
  __int64 *v64; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v65; // [rsp+58h] [rbp-A8h]
  _BYTE v66[16]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v67; // [rsp+70h] [rbp-90h] BYREF
  __int64 v68; // [rsp+78h] [rbp-88h]
  __int64 v69; // [rsp+80h] [rbp-80h]
  unsigned int v70; // [rsp+88h] [rbp-78h]
  __m128i v71; // [rsp+90h] [rbp-70h] BYREF
  __m128i v72; // [rsp+A0h] [rbp-60h] BYREF
  __m128i *v73; // [rsp+B0h] [rbp-50h] BYREF
  __m128i *v74; // [rsp+B8h] [rbp-48h]
  _QWORD v75[8]; // [rsp+C0h] [rbp-40h] BYREF

  v5 = a2;
  v6 = a1;
  v7 = *(_BYTE *)(a1 + 48) == 0;
  v64 = (__int64 *)v66;
  v65 = 0x200000000LL;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  if ( !v7 )
  {
    v59 = &a3[a4];
    if ( v59 != a3 )
    {
      v9 = a3;
      while ( 1 )
      {
        v13 = *v9;
        v14 = sub_13774B0(*(_QWORD *)(a1 + 40), *v9, a2);
        v73 = (__m128i *)sub_1368AA0(*(__int64 **)(a1 + 32), v13);
        v15 = sub_16AF500((__int64 *)&v73, v14);
        if ( !v70 )
          break;
        v10 = (v70 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v11 = (__int64 *)(v68 + 16LL * v10);
        v12 = *v11;
        if ( v13 != *v11 )
        {
          v47 = 1;
          v18 = 0;
          while ( v12 != -8 )
          {
            if ( v12 == -16 && !v18 )
              v18 = v11;
            v10 = (v70 - 1) & (v47 + v10);
            v11 = (__int64 *)(v68 + 16LL * v10);
            v12 = *v11;
            if ( v13 == *v11 )
              goto LABEL_5;
            ++v47;
          }
          if ( !v18 )
            v18 = v11;
          ++v67;
          v17 = v69 + 1;
          if ( 4 * ((int)v69 + 1) < 3 * v70 )
          {
            if ( v70 - HIDWORD(v69) - v17 <= v70 >> 3 )
            {
              sub_1956860((__int64)&v67, v70);
              if ( !v70 )
              {
LABEL_111:
                LODWORD(v69) = v69 + 1;
                BUG();
              }
              v50 = 0;
              LODWORD(v51) = (v70 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
              v52 = 1;
              v17 = v69 + 1;
              v18 = (__int64 *)(v68 + 16LL * (unsigned int)v51);
              v53 = *v18;
              if ( v13 != *v18 )
              {
                while ( v53 != -8 )
                {
                  if ( !v50 && v53 == -16 )
                    v50 = v18;
                  v51 = (v70 - 1) & ((_DWORD)v51 + v52);
                  v18 = (__int64 *)(v68 + 16 * v51);
                  v53 = *v18;
                  if ( v13 == *v18 )
                    goto LABEL_77;
                  ++v52;
                }
                if ( v50 )
                  v18 = v50;
              }
            }
            goto LABEL_77;
          }
LABEL_8:
          sub_1956860((__int64)&v67, 2 * v70);
          if ( !v70 )
            goto LABEL_111;
          LODWORD(v16) = (v70 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v17 = v69 + 1;
          v18 = (__int64 *)(v68 + 16LL * (unsigned int)v16);
          v19 = *v18;
          if ( v13 != *v18 )
          {
            v20 = 1;
            v21 = 0;
            while ( v19 != -8 )
            {
              if ( v19 == -16 && !v21 )
                v21 = v18;
              v16 = (v70 - 1) & ((_DWORD)v16 + v20);
              v18 = (__int64 *)(v68 + 16 * v16);
              v19 = *v18;
              if ( v13 == *v18 )
                goto LABEL_77;
              ++v20;
            }
            if ( v21 )
              v18 = v21;
          }
LABEL_77:
          LODWORD(v69) = v17;
          if ( *v18 != -8 )
            --HIDWORD(v69);
          *v18 = v13;
          v18[1] = v15;
        }
LABEL_5:
        if ( v59 == ++v9 )
        {
          v6 = a1;
          v5 = a2;
          goto LABEL_15;
        }
      }
      ++v67;
      goto LABEL_8;
    }
  }
LABEL_15:
  if ( sub_157F790(v5) )
  {
    v73 = (__m128i *)v75;
    if ( !a5 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v22 = (const __m128i *)strlen(a5);
    v71.m128i_i64[0] = (__int64)v22;
    v23 = (size_t)v22;
    if ( (unsigned __int64)v22 > 0xF )
    {
      v73 = (__m128i *)sub_22409D0(&v73, &v71, 0);
      v48 = v73;
      v75[0] = v71.m128i_i64[0];
    }
    else
    {
      if ( v22 == (const __m128i *)1 )
      {
        LOBYTE(v75[0]) = *a5;
        v24 = (const __m128i *)v75;
LABEL_20:
        v74 = (__m128i *)v22;
        v22->m128i_i8[(_QWORD)v24] = 0;
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - (_QWORD)v74) <= 8 )
          sub_4262D8((__int64)"basic_string::append");
        v25 = (__m128i *)sub_2241490(&v73, ".split-lp", 9);
        v71.m128i_i64[0] = (__int64)&v72;
        if ( (__m128i *)v25->m128i_i64[0] == &v25[1] )
        {
          v72 = _mm_loadu_si128(v25 + 1);
        }
        else
        {
          v71.m128i_i64[0] = v25->m128i_i64[0];
          v72.m128i_i64[0] = v25[1].m128i_i64[0];
        }
        v71.m128i_i64[1] = v25->m128i_i64[1];
        v25->m128i_i64[0] = (__int64)v25[1].m128i_i64;
        v25->m128i_i64[1] = 0;
        v25[1].m128i_i8[0] = 0;
        if ( v73 != (__m128i *)v75 )
          j_j___libc_free_0(v73, v75[0] + 1LL);
        sub_1AAA850(v5, (_DWORD)a3, a4, (_DWORD)a5, v71.m128i_i32[0], (unsigned int)&v64, 0, 0, 0);
        if ( (__m128i *)v71.m128i_i64[0] != &v72 )
          j_j___libc_free_0(v71.m128i_i64[0], v72.m128i_i64[0] + 1);
        v26 = (unsigned int)v65;
        goto LABEL_31;
      }
      if ( !v22 )
      {
        v24 = (const __m128i *)v75;
        goto LABEL_20;
      }
      v48 = (__m128i *)v75;
    }
    memcpy(v48, a5, v23);
    v22 = (const __m128i *)v71.m128i_i64[0];
    v24 = v73;
    goto LABEL_20;
  }
  v27 = sub_1AAB350(v5, a3, a4, a5, 0, 0, 0);
  v30 = (unsigned int)v65;
  if ( (unsigned int)v65 >= HIDWORD(v65) )
  {
    sub_16CD150((__int64)&v64, v66, 0, 8, v28, v29);
    v30 = (unsigned int)v65;
  }
  v64[v30] = v27;
  v26 = (unsigned int)(v65 + 1);
  LODWORD(v65) = v65 + 1;
LABEL_31:
  v73 = 0;
  v74 = 0;
  v75[0] = 0;
  sub_1953AE0((const __m128i **)&v73, v26 + 2 * a4);
  sa = v5 & 0xFFFFFFFFFFFFFFFBLL;
  v62 = v64;
  v55 = &v64[(unsigned int)v65];
  if ( v64 != v55 )
  {
LABEL_32:
    v31 = v74;
    v32 = *v62;
    v63 = 0;
    v60 = v32;
    v71.m128i_i64[0] = v32;
    v71.m128i_i64[1] = sa;
    if ( v74 == (__m128i *)v75[0] )
    {
      sub_17F2860((const __m128i **)&v73, v74, &v71);
    }
    else
    {
      if ( v74 )
      {
        *v74 = _mm_load_si128(&v71);
        v31 = v74;
      }
      v74 = (__m128i *)&v31[1];
    }
    v33 = *(_QWORD *)(v60 + 8);
    if ( !v33 )
    {
LABEL_52:
      if ( *(_BYTE *)(v6 + 48) )
        goto LABEL_61;
      goto LABEL_53;
    }
    while ( 1 )
    {
      v34 = sub_1648700(v33);
      if ( (unsigned __int8)(*((_BYTE *)v34 + 16) - 25) <= 9u )
        break;
      v33 = *(_QWORD *)(v33 + 8);
      if ( !v33 )
      {
        if ( *(_BYTE *)(v6 + 48) )
LABEL_61:
          sub_136C010(*(__int64 **)(v6 + 32), v60, v63);
LABEL_53:
        if ( v55 == ++v62 )
          goto LABEL_54;
        goto LABEL_32;
      }
    }
    v35 = v33;
    while ( 1 )
    {
      v36 = v34[5];
      v37 = v74;
      v71.m128i_i64[1] = sa | 4;
      v38 = (__m128i *)v75[0];
      v71.m128i_i64[0] = v36;
      if ( v74 == (__m128i *)v75[0] )
      {
        sub_17F2860((const __m128i **)&v73, v74, &v71);
        v71.m128i_i64[0] = v36;
        v39 = v74;
        v71.m128i_i64[1] = v60 & 0xFFFFFFFFFFFFFFFBLL;
        if ( v74 != (__m128i *)v75[0] )
        {
          if ( v74 )
          {
LABEL_45:
            *v39 = _mm_load_si128(&v71);
            v39 = v74;
          }
          v74 = v39 + 1;
          goto LABEL_47;
        }
      }
      else
      {
        if ( v74 )
        {
          *v74 = _mm_load_si128(&v71);
          v37 = v74;
          v38 = (__m128i *)v75[0];
        }
        v39 = v37 + 1;
        v71.m128i_i64[0] = v36;
        v74 = v39;
        v71.m128i_i64[1] = v60 & 0xFFFFFFFFFFFFFFFBLL;
        if ( v38 != v39 )
          goto LABEL_45;
      }
      sub_17F2860((const __m128i **)&v73, v39, &v71);
LABEL_47:
      if ( *(_BYTE *)(v6 + 48) )
      {
        v40 = 0;
        if ( v70 )
        {
          v41 = (v70 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
          v42 = (__int64 *)(v68 + 16LL * v41);
          v43 = *v42;
          if ( v36 == *v42 )
          {
LABEL_50:
            v40 = v42[1];
          }
          else
          {
            v46 = 1;
            while ( v43 != -8 )
            {
              v49 = v46 + 1;
              v41 = (v70 - 1) & (v46 + v41);
              v42 = (__int64 *)(v68 + 16LL * v41);
              v43 = *v42;
              if ( v36 == *v42 )
                goto LABEL_50;
              v46 = v49;
            }
            v40 = 0;
          }
        }
        sub_16AF570(&v63, v40);
        v35 = *(_QWORD *)(v35 + 8);
        if ( !v35 )
          goto LABEL_52;
        goto LABEL_40;
      }
      do
      {
        v35 = *(_QWORD *)(v35 + 8);
        if ( !v35 )
          goto LABEL_52;
LABEL_40:
        v34 = sub_1648700(v35);
      }
      while ( (unsigned __int8)(*((_BYTE *)v34 + 16) - 25) > 9u );
    }
  }
LABEL_54:
  sub_15CD9D0(*(_QWORD *)(v6 + 24), v73->m128i_i64, v74 - v73);
  v44 = *v64;
  if ( v73 )
    j_j___libc_free_0(v73, v75[0] - (_QWORD)v73);
  j___libc_free_0(v68);
  if ( v64 != (__int64 *)v66 )
    _libc_free((unsigned __int64)v64);
  return v44;
}
