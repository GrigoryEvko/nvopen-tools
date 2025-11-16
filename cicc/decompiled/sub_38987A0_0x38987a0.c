// Function: sub_38987A0
// Address: 0x38987a0
//
__int64 *__fastcall sub_38987A0(_QWORD *a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 v4; // r14
  _QWORD *v5; // rbx
  __int64 v6; // r13
  __int64 v8; // r15
  int v9; // eax
  __int64 v10; // rax
  __int64 *v11; // r15
  __m128i *v13; // rax
  size_t v14; // rcx
  __m128i *v15; // r9
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdi
  __m128i *v18; // rax
  unsigned __int64 v19; // rcx
  __m128i *v20; // rdx
  __m128i *v21; // rax
  __int64 v22; // rax
  bool v23; // zf
  __int64 v24; // rax
  __int64 v25; // rbx
  unsigned __int8 *v26; // r13
  size_t v27; // r14
  _QWORD *v28; // r12
  size_t v29; // r15
  size_t v30; // rdx
  int v31; // eax
  unsigned __int8 *v32; // r11
  size_t v33; // r8
  _QWORD *v34; // r9
  size_t v35; // r12
  size_t v36; // rdx
  int v37; // eax
  __int64 v38; // r8
  __int64 v39; // rax
  _BYTE *v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // r12
  __int64 v43; // rax
  _QWORD *v44; // rdx
  unsigned __int64 v45; // r9
  __int64 v46; // r13
  _QWORD *v47; // r12
  unsigned int v48; // edi
  _QWORD *v49; // rcx
  unsigned int v50; // r12d
  unsigned __int64 v51; // rdi
  size_t v52; // r13
  size_t v53; // rcx
  size_t v54; // rdx
  int v55; // eax
  unsigned int v56; // edi
  __int64 v57; // r13
  unsigned __int64 v58; // [rsp+8h] [rbp-118h]
  _QWORD *v59; // [rsp+10h] [rbp-110h]
  __int64 v60; // [rsp+18h] [rbp-108h]
  size_t v61; // [rsp+18h] [rbp-108h]
  __int64 v62; // [rsp+18h] [rbp-108h]
  size_t v63; // [rsp+18h] [rbp-108h]
  __int64 v64; // [rsp+20h] [rbp-100h]
  __int64 *v65; // [rsp+20h] [rbp-100h]
  _QWORD *v66; // [rsp+20h] [rbp-100h]
  unsigned __int64 v67; // [rsp+20h] [rbp-100h]
  _QWORD *v68; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v69; // [rsp+28h] [rbp-F8h]
  unsigned __int64 *v70; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v71; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v72[2]; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v73; // [rsp+60h] [rbp-C0h] BYREF
  __m128i *v74; // [rsp+70h] [rbp-B0h] BYREF
  size_t v75; // [rsp+78h] [rbp-A8h]
  __m128i v76; // [rsp+80h] [rbp-A0h] BYREF
  char *v77; // [rsp+90h] [rbp-90h] BYREF
  size_t v78; // [rsp+98h] [rbp-88h]
  _QWORD v79[2]; // [rsp+A0h] [rbp-80h] BYREF
  __m128i *v80; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v81; // [rsp+B8h] [rbp-68h]
  __m128i v82; // [rsp+C0h] [rbp-60h] BYREF
  unsigned __int64 v83[2]; // [rsp+D0h] [rbp-50h] BYREF
  _OWORD v84[4]; // [rsp+E0h] [rbp-40h] BYREF

  v4 = a4;
  v5 = a1;
  if ( *(_BYTE *)(a3 + 8) != 15 )
  {
    LOWORD(v84[0]) = 259;
    v11 = 0;
    v83[0] = (unsigned __int64)"global variable reference must have pointer type";
    sub_38814C0((__int64)(a1 + 1), a4, (__int64)v83);
    return v11;
  }
  v6 = a2;
  v8 = *(_QWORD *)(a1[22] + 120LL);
  v9 = sub_16D1B30((__int64 *)v8, *(unsigned __int8 **)a2, *(_QWORD *)(a2 + 8));
  if ( v9 != -1
    && (v10 = *(_QWORD *)v8 + 8LL * v9, v10 != *(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8))
    && (v11 = *(__int64 **)(*(_QWORD *)v10 + 8LL)) != 0
    || (v22 = sub_38902E0((__int64)(a1 + 113), a2), v68 = a1 + 114, (_QWORD *)v22 != a1 + 114)
    && (v11 = *(__int64 **)(v22 + 64)) != 0 )
  {
    if ( a3 == *v11 )
      return v11;
    sub_3888960((__int64 *)&v77, *v11);
    sub_8FD6D0((__int64)v72, "'@", (_QWORD *)a2);
    if ( 0x3FFFFFFFFFFFFFFFLL - v72[1] <= 0x14 )
      goto LABEL_85;
    v13 = (__m128i *)sub_2241490(v72, "' defined with type '", 0x15u);
    v74 = &v76;
    if ( (__m128i *)v13->m128i_i64[0] == &v13[1] )
    {
      v76 = _mm_loadu_si128(v13 + 1);
    }
    else
    {
      v74 = (__m128i *)v13->m128i_i64[0];
      v76.m128i_i64[0] = v13[1].m128i_i64[0];
    }
    v14 = v13->m128i_u64[1];
    v13[1].m128i_i8[0] = 0;
    v75 = v14;
    v13->m128i_i64[0] = (__int64)v13[1].m128i_i64;
    v15 = v74;
    v13->m128i_i64[1] = 0;
    v16 = 15;
    v17 = 15;
    if ( v15 != &v76 )
      v17 = v76.m128i_i64[0];
    if ( v75 + v78 <= v17 )
      goto LABEL_17;
    if ( v77 != (char *)v79 )
      v16 = v79[0];
    if ( v75 + v78 <= v16 )
    {
      v18 = (__m128i *)sub_2241130((unsigned __int64 *)&v77, 0, 0, v15, v75);
      v80 = &v82;
      v19 = v18->m128i_i64[0];
      v20 = v18 + 1;
      if ( (__m128i *)v18->m128i_i64[0] != &v18[1] )
        goto LABEL_18;
    }
    else
    {
LABEL_17:
      v18 = (__m128i *)sub_2241490((unsigned __int64 *)&v74, v77, v78);
      v80 = &v82;
      v19 = v18->m128i_i64[0];
      v20 = v18 + 1;
      if ( (__m128i *)v18->m128i_i64[0] != &v18[1] )
      {
LABEL_18:
        v80 = (__m128i *)v19;
        v82.m128i_i64[0] = v18[1].m128i_i64[0];
        goto LABEL_19;
      }
    }
    v82 = _mm_loadu_si128(v18 + 1);
LABEL_19:
    v81 = v18->m128i_i64[1];
    v18->m128i_i64[0] = (__int64)v20;
    v18->m128i_i64[1] = 0;
    v18[1].m128i_i8[0] = 0;
    if ( v81 != 0x3FFFFFFFFFFFFFFFLL )
    {
      v21 = (__m128i *)sub_2241490((unsigned __int64 *)&v80, "'", 1u);
      v83[0] = (unsigned __int64)v84;
      if ( (__m128i *)v21->m128i_i64[0] == &v21[1] )
      {
        v84[0] = _mm_loadu_si128(v21 + 1);
      }
      else
      {
        v83[0] = v21->m128i_i64[0];
        *(_QWORD *)&v84[0] = v21[1].m128i_i64[0];
      }
      v83[1] = v21->m128i_u64[1];
      v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
      v21->m128i_i64[1] = 0;
      v21[1].m128i_i8[0] = 0;
      v71 = 260;
      v70 = v83;
      sub_38814C0((__int64)(v5 + 1), v4, (__int64)&v70);
      if ( (_OWORD *)v83[0] != v84 )
        j_j___libc_free_0(v83[0]);
      if ( v80 != &v82 )
        j_j___libc_free_0((unsigned __int64)v80);
      if ( v74 != &v76 )
        j_j___libc_free_0((unsigned __int64)v74);
      if ( (__int64 *)v72[0] != &v73 )
        j_j___libc_free_0(v72[0]);
      if ( v77 != (char *)v79 )
        j_j___libc_free_0((unsigned __int64)v77);
      return 0;
    }
LABEL_85:
    sub_4262D8((__int64)"basic_string::append");
  }
  v23 = *(_BYTE *)(*(_QWORD *)(a3 + 24) + 8LL) == 12;
  v60 = *(_QWORD *)(a3 + 24);
  v64 = a1[22];
  LOWORD(v84[0]) = 260;
  v83[0] = a2;
  if ( v23 )
  {
    v24 = sub_1648B60(120);
    v11 = (__int64 *)v24;
    if ( v24 )
      sub_15E2490(v24, v60, 9, (__int64)v83, v64);
  }
  else
  {
    v50 = *(_DWORD *)(a3 + 8) >> 8;
    v11 = sub_1648A60(88, 1u);
    if ( v11 )
      sub_15E51E0((__int64)v11, v64, v60, 0, 9, 0, (__int64)v83, 0, 0, v50, 0);
  }
  if ( !a1[115] )
  {
    v34 = a1 + 114;
    goto LABEL_60;
  }
  v25 = a1[115];
  v58 = v4;
  v26 = *(unsigned __int8 **)a2;
  v27 = *(_QWORD *)(a2 + 8);
  v65 = v11;
  v28 = a1 + 114;
  do
  {
    v29 = *(_QWORD *)(v25 + 40);
    v30 = v27;
    if ( v29 <= v27 )
      v30 = *(_QWORD *)(v25 + 40);
    if ( !v30 || (v31 = memcmp(*(const void **)(v25 + 32), v26, v30)) == 0 )
    {
      if ( (__int64)(v29 - v27) >= 0x80000000LL )
        goto LABEL_50;
      if ( (__int64)(v29 - v27) <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_40;
      v31 = v29 - v27;
    }
    if ( v31 < 0 )
    {
LABEL_40:
      v25 = *(_QWORD *)(v25 + 24);
      continue;
    }
LABEL_50:
    v28 = (_QWORD *)v25;
    v25 = *(_QWORD *)(v25 + 16);
  }
  while ( v25 );
  v32 = v26;
  v33 = v27;
  v11 = v65;
  v34 = v28;
  v5 = a1;
  v6 = a2;
  v4 = v58;
  if ( v68 == v28 )
    goto LABEL_60;
  v35 = v28[5];
  v36 = v33;
  if ( v35 <= v33 )
    v36 = v35;
  if ( v36 && (v61 = v33, v66 = v34, v37 = memcmp(v32, (const void *)v34[4], v36), v34 = v66, v33 = v61, v37) )
  {
LABEL_59:
    if ( v37 < 0 )
      goto LABEL_60;
  }
  else
  {
    v38 = v33 - v35;
    if ( v38 <= 0x7FFFFFFF )
    {
      if ( v38 >= (__int64)0xFFFFFFFF80000000LL )
      {
        v37 = v38;
        goto LABEL_59;
      }
LABEL_60:
      v59 = v34;
      v39 = sub_22077B0(0x50u);
      v40 = *(_BYTE **)v6;
      v41 = *(_QWORD *)(v6 + 8);
      v42 = v39 + 32;
      v67 = v39;
      *(_QWORD *)(v39 + 32) = v39 + 48;
      v62 = v39 + 48;
      sub_3887850((__int64 *)(v39 + 32), v40, (__int64)&v40[v41]);
      *(_QWORD *)(v67 + 64) = 0;
      *(_QWORD *)(v67 + 72) = 0;
      v43 = sub_3898510(a1 + 113, v59, v42);
      v45 = v67;
      v46 = v43;
      v47 = v44;
      if ( v44 )
      {
        if ( v43 || v68 == v44 )
        {
LABEL_63:
          LOBYTE(v48) = 1;
          goto LABEL_64;
        }
        v52 = *(_QWORD *)(v67 + 40);
        v54 = v44[5];
        v53 = v54;
        if ( v52 <= v54 )
          v54 = *(_QWORD *)(v67 + 40);
        if ( v54
          && (v63 = v53,
              v55 = memcmp(*(const void **)(v67 + 32), (const void *)v47[4], v54),
              v45 = v67,
              v53 = v63,
              (v56 = v55) != 0) )
        {
LABEL_84:
          v48 = v56 >> 31;
        }
        else
        {
          v57 = v52 - v53;
          LOBYTE(v48) = 0;
          if ( v57 <= 0x7FFFFFFF )
          {
            if ( v57 < (__int64)0xFFFFFFFF80000000LL )
              goto LABEL_63;
            v56 = v57;
            goto LABEL_84;
          }
        }
LABEL_64:
        v49 = v68;
        v69 = v45;
        sub_220F040(v48, v45, v47, v49);
        ++v5[118];
        v34 = (_QWORD *)v69;
      }
      else
      {
        v51 = *(_QWORD *)(v67 + 32);
        if ( v62 != v51 )
        {
          j_j___libc_free_0(v51);
          v45 = v67;
        }
        j_j___libc_free_0(v45);
        v34 = (_QWORD *)v46;
      }
    }
  }
  v34[8] = v11;
  v34[9] = v4;
  return v11;
}
