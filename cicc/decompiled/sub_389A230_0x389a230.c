// Function: sub_389A230
// Address: 0x389a230
//
__int64 *__fastcall sub_389A230(_QWORD *a1, __int64 a2, __int64 a3, unsigned __int64 a4, char a5)
{
  __int64 v5; // r14
  _QWORD *v7; // r12
  __int64 v9; // r15
  int v10; // eax
  __int64 v11; // rax
  __int64 *v12; // r15
  char v13; // al
  __m128i *v14; // rax
  size_t v15; // rcx
  __m128i *v16; // r9
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdi
  __m128i *v19; // rax
  unsigned __int64 v20; // rcx
  __m128i *v21; // rdx
  __m128i *v22; // rax
  char *v23; // rdi
  __int64 v24; // rax
  char v25; // dl
  __int64 v26; // rdi
  __int64 *v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r15
  unsigned __int8 *v32; // r12
  size_t v33; // r14
  _QWORD *v34; // r13
  size_t v35; // rbx
  size_t v36; // rdx
  int v37; // eax
  __int64 v38; // rbx
  unsigned __int8 *v39; // r11
  size_t v40; // rcx
  _QWORD *v41; // r8
  size_t v42; // rbx
  size_t v43; // rdx
  int v44; // eax
  __int64 v45; // rcx
  __int64 v46; // rax
  _BYTE *v47; // rsi
  __int64 v48; // rdx
  __int64 v49; // rbx
  __int64 v50; // r13
  __int64 v51; // rax
  _QWORD *v52; // rdx
  unsigned __int64 v53; // r8
  __int64 v54; // r14
  _QWORD *v55; // r13
  unsigned int v56; // edi
  _QWORD *v57; // rcx
  __m128i *v58; // rax
  __int64 v59; // rbx
  __int64 v60; // r13
  __int64 *v61; // rax
  unsigned __int64 v62; // rdi
  size_t v63; // rbx
  size_t v64; // r14
  size_t v65; // rdx
  int v66; // eax
  unsigned int v67; // edi
  __int64 v68; // rbx
  size_t v69; // [rsp+8h] [rbp-118h]
  _QWORD *v70; // [rsp+8h] [rbp-118h]
  __int64 *v71; // [rsp+10h] [rbp-110h]
  unsigned __int64 v72; // [rsp+10h] [rbp-110h]
  __int64 v73; // [rsp+20h] [rbp-100h]
  _QWORD *v74; // [rsp+20h] [rbp-100h]
  unsigned __int64 v75; // [rsp+20h] [rbp-100h]
  unsigned __int64 v77[2]; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v78; // [rsp+40h] [rbp-E0h] BYREF
  unsigned __int64 v79[2]; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v80; // [rsp+60h] [rbp-C0h] BYREF
  __m128i *v81; // [rsp+70h] [rbp-B0h] BYREF
  size_t v82; // [rsp+78h] [rbp-A8h]
  __m128i v83; // [rsp+80h] [rbp-A0h] BYREF
  char *v84; // [rsp+90h] [rbp-90h] BYREF
  size_t v85; // [rsp+98h] [rbp-88h]
  _QWORD v86[2]; // [rsp+A0h] [rbp-80h] BYREF
  const char **v87; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v88; // [rsp+B8h] [rbp-68h]
  __m128i v89; // [rsp+C0h] [rbp-60h] BYREF
  const char *v90; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v91; // [rsp+D8h] [rbp-48h]
  _OWORD v92[4]; // [rsp+E0h] [rbp-40h] BYREF

  v5 = a2;
  v7 = a1;
  v9 = *(_QWORD *)(a1[1] + 104LL);
  v10 = sub_16D1B30((__int64 *)v9, *(unsigned __int8 **)a2, *(_QWORD *)(a2 + 8));
  if ( v10 != -1
    && (v11 = *(_QWORD *)v9 + 8LL * v10, v11 != *(_QWORD *)v9 + 8LL * *(unsigned int *)(v9 + 8))
    && (v12 = *(__int64 **)(*(_QWORD *)v11 + 8LL)) != 0
    || (v24 = sub_3890600((__int64)(a1 + 2), a2), v74 = a1 + 3, (_QWORD *)v24 != a1 + 3)
    && (v12 = *(__int64 **)(v24 + 64)) != 0 )
  {
    if ( a3 == *v12 )
      return v12;
    v73 = *a1;
    v13 = *(_BYTE *)(a3 + 8);
    if ( a5 && v13 == 15 )
    {
      v28 = *(__int64 **)(a3 + 24);
      v29 = sub_1632FA0(*(_QWORD *)(v73 + 176));
      if ( *v12 == sub_1647190(v28, *(_DWORD *)(v29 + 12)) )
        return v12;
      v73 = *a1;
      v13 = *(_BYTE *)(a3 + 8);
    }
    if ( v13 == 7 )
    {
      sub_8FD6D0((__int64)v77, "'%", (_QWORD *)a2);
      if ( 0x3FFFFFFFFFFFFFFFLL - v77[1] > 0x15 )
      {
        v58 = (__m128i *)sub_2241490(v77, "' is not a basic block", 0x16u);
        v90 = (const char *)v92;
        if ( (__m128i *)v58->m128i_i64[0] == &v58[1] )
        {
          v92[0] = _mm_loadu_si128(v58 + 1);
        }
        else
        {
          v90 = (const char *)v58->m128i_i64[0];
          *(_QWORD *)&v92[0] = v58[1].m128i_i64[0];
        }
        v91 = v58->m128i_i64[1];
        v58->m128i_i64[0] = (__int64)v58[1].m128i_i64;
        v58->m128i_i64[1] = 0;
        v58[1].m128i_i8[0] = 0;
        v89.m128i_i16[0] = 260;
        v87 = &v90;
        sub_38814C0(v73 + 8, a4, (__int64)&v87);
        if ( v90 != (const char *)v92 )
          j_j___libc_free_0((unsigned __int64)v90);
        v23 = (char *)v77[0];
        if ( (__int64 *)v77[0] == &v78 )
          return 0;
LABEL_30:
        j_j___libc_free_0((unsigned __int64)v23);
        return 0;
      }
LABEL_98:
      sub_4262D8((__int64)"basic_string::append");
    }
    sub_3888960((__int64 *)&v84, *v12);
    sub_8FD6D0((__int64)v79, "'%", (_QWORD *)a2);
    if ( 0x3FFFFFFFFFFFFFFFLL - v79[1] <= 0x14 )
      goto LABEL_98;
    v14 = (__m128i *)sub_2241490(v79, "' defined with type '", 0x15u);
    v81 = &v83;
    if ( (__m128i *)v14->m128i_i64[0] == &v14[1] )
    {
      v83 = _mm_loadu_si128(v14 + 1);
    }
    else
    {
      v81 = (__m128i *)v14->m128i_i64[0];
      v83.m128i_i64[0] = v14[1].m128i_i64[0];
    }
    v15 = v14->m128i_u64[1];
    v14[1].m128i_i8[0] = 0;
    v82 = v15;
    v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
    v16 = v81;
    v14->m128i_i64[1] = 0;
    v17 = 15;
    v18 = 15;
    if ( v16 != &v83 )
      v18 = v83.m128i_i64[0];
    if ( v82 + v85 <= v18 )
      goto LABEL_16;
    if ( v84 != (char *)v86 )
      v17 = v86[0];
    if ( v82 + v85 <= v17 )
    {
      v19 = (__m128i *)sub_2241130((unsigned __int64 *)&v84, 0, 0, v16, v82);
      v87 = (const char **)&v89;
      v20 = v19->m128i_i64[0];
      v21 = v19 + 1;
      if ( (__m128i *)v19->m128i_i64[0] != &v19[1] )
        goto LABEL_17;
    }
    else
    {
LABEL_16:
      v19 = (__m128i *)sub_2241490((unsigned __int64 *)&v81, v84, v85);
      v87 = (const char **)&v89;
      v20 = v19->m128i_i64[0];
      v21 = v19 + 1;
      if ( (__m128i *)v19->m128i_i64[0] != &v19[1] )
      {
LABEL_17:
        v87 = (const char **)v20;
        v89.m128i_i64[0] = v19[1].m128i_i64[0];
        goto LABEL_18;
      }
    }
    v89 = _mm_loadu_si128(v19 + 1);
LABEL_18:
    v88 = v19->m128i_i64[1];
    v19->m128i_i64[0] = (__int64)v21;
    v19->m128i_i64[1] = 0;
    v19[1].m128i_i8[0] = 0;
    if ( v88 != 0x3FFFFFFFFFFFFFFFLL )
    {
      v22 = (__m128i *)sub_2241490((unsigned __int64 *)&v87, "'", 1u);
      v90 = (const char *)v92;
      if ( (__m128i *)v22->m128i_i64[0] == &v22[1] )
      {
        v92[0] = _mm_loadu_si128(v22 + 1);
      }
      else
      {
        v90 = (const char *)v22->m128i_i64[0];
        *(_QWORD *)&v92[0] = v22[1].m128i_i64[0];
      }
      v91 = v22->m128i_i64[1];
      v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
      v22->m128i_i64[1] = 0;
      v22[1].m128i_i8[0] = 0;
      LOWORD(v78) = 260;
      v77[0] = (unsigned __int64)&v90;
      sub_38814C0(v73 + 8, a4, (__int64)v77);
      if ( v90 != (const char *)v92 )
        j_j___libc_free_0((unsigned __int64)v90);
      if ( v87 != (const char **)&v89 )
        j_j___libc_free_0((unsigned __int64)v87);
      if ( v81 != &v83 )
        j_j___libc_free_0((unsigned __int64)v81);
      if ( (__int64 *)v79[0] != &v80 )
        j_j___libc_free_0(v79[0]);
      v23 = v84;
      if ( v84 == (char *)v86 )
        return 0;
      goto LABEL_30;
    }
    goto LABEL_98;
  }
  v25 = *(_BYTE *)(a3 + 8);
  if ( v25 == 12 || !*(_BYTE *)(a3 + 8) )
  {
    v26 = *a1;
    LOWORD(v92[0]) = 259;
    v90 = "invalid use of a non-first-class type";
    sub_38814C0(v26 + 8, a4, (__int64)&v90);
    return 0;
  }
  if ( v25 == 7 )
  {
    v59 = a1[1];
    v90 = (const char *)a2;
    LOWORD(v92[0]) = 260;
    v60 = sub_15E0530(v59);
    v61 = (__int64 *)sub_22077B0(0x40u);
    v12 = v61;
    if ( v61 )
      sub_157FB60(v61, v60, (__int64)&v90, v59, 0);
  }
  else
  {
    v90 = (const char *)a2;
    LOWORD(v92[0]) = 260;
    v30 = sub_22077B0(0x28u);
    v12 = (__int64 *)v30;
    if ( v30 )
      sub_15E0280(v30, a3, (__int64)&v90, 0, 0);
  }
  if ( !a1[4] )
  {
    v41 = a1 + 3;
    goto LABEL_65;
  }
  v71 = v12;
  v31 = a1[4];
  v32 = *(unsigned __int8 **)a2;
  v33 = *(_QWORD *)(a2 + 8);
  v34 = a1 + 3;
  do
  {
    while ( 1 )
    {
      v35 = *(_QWORD *)(v31 + 40);
      v36 = v33;
      if ( v35 <= v33 )
        v36 = *(_QWORD *)(v31 + 40);
      if ( v36 )
      {
        v37 = memcmp(*(const void **)(v31 + 32), v32, v36);
        if ( v37 )
          break;
      }
      v38 = v35 - v33;
      if ( v38 >= 0x80000000LL )
        goto LABEL_55;
      if ( v38 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v37 = v38;
        break;
      }
LABEL_46:
      v31 = *(_QWORD *)(v31 + 24);
      if ( !v31 )
        goto LABEL_56;
    }
    if ( v37 < 0 )
      goto LABEL_46;
LABEL_55:
    v34 = (_QWORD *)v31;
    v31 = *(_QWORD *)(v31 + 16);
  }
  while ( v31 );
LABEL_56:
  v39 = v32;
  v40 = v33;
  v12 = v71;
  v41 = v34;
  v7 = a1;
  v5 = a2;
  if ( v74 == v34 )
    goto LABEL_65;
  v42 = v34[5];
  v43 = v40;
  if ( v42 <= v40 )
    v43 = v34[5];
  if ( v43 && (v69 = v40, v44 = memcmp(v39, (const void *)v34[4], v43), v40 = v69, v41 = v34, v44) )
  {
LABEL_64:
    if ( v44 < 0 )
      goto LABEL_65;
  }
  else
  {
    v45 = v40 - v42;
    if ( v45 <= 0x7FFFFFFF )
    {
      if ( v45 >= (__int64)0xFFFFFFFF80000000LL )
      {
        v44 = v45;
        goto LABEL_64;
      }
LABEL_65:
      v70 = v41;
      v46 = sub_22077B0(0x50u);
      v47 = *(_BYTE **)v5;
      v48 = *(_QWORD *)(v5 + 8);
      v49 = v46 + 48;
      v50 = v46 + 32;
      v72 = v46;
      *(_QWORD *)(v46 + 32) = v46 + 48;
      sub_3887850((__int64 *)(v46 + 32), v47, (__int64)&v47[v48]);
      *(_QWORD *)(v72 + 64) = 0;
      *(_QWORD *)(v72 + 72) = 0;
      v51 = sub_3899FA0(a1 + 2, v70, v50);
      v53 = v72;
      v54 = v51;
      v55 = v52;
      if ( v52 )
      {
        if ( v74 == v52 || v51 )
        {
LABEL_68:
          LOBYTE(v56) = 1;
          goto LABEL_69;
        }
        v63 = *(_QWORD *)(v72 + 40);
        v65 = v52[5];
        v64 = v65;
        if ( v63 <= v65 )
          v65 = *(_QWORD *)(v72 + 40);
        if ( v65 && (v66 = memcmp(*(const void **)(v72 + 32), (const void *)v55[4], v65), v53 = v72, (v67 = v66) != 0) )
        {
LABEL_97:
          v56 = v67 >> 31;
        }
        else
        {
          v68 = v63 - v64;
          LOBYTE(v56) = 0;
          if ( v68 <= 0x7FFFFFFF )
          {
            if ( v68 < (__int64)0xFFFFFFFF80000000LL )
              goto LABEL_68;
            v67 = v68;
            goto LABEL_97;
          }
        }
LABEL_69:
        v57 = v74;
        v75 = v53;
        sub_220F040(v56, v53, v55, v57);
        ++v7[7];
        v41 = (_QWORD *)v75;
      }
      else
      {
        v62 = *(_QWORD *)(v72 + 32);
        if ( v49 != v62 )
        {
          j_j___libc_free_0(v62);
          v53 = v72;
        }
        j_j___libc_free_0(v53);
        v41 = (_QWORD *)v54;
      }
    }
  }
  v41[8] = v12;
  v41[9] = a4;
  return v12;
}
