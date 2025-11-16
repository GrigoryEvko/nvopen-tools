// Function: sub_26FC690
// Address: 0x26fc690
//
__int64 __fastcall sub_26FC690(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned __int8 *v10; // r13
  unsigned int v11; // r12d
  size_t v13; // rdx
  const char *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
  const void *v17; // rax
  size_t v18; // rdx
  __m128i v19; // xmm1
  _BYTE *v20; // r15
  _BYTE *v21; // r14
  __m128i *v22; // rdi
  __int64 (__fastcall *v23)(__int64); // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rdi
  size_t *v28; // r15
  size_t *v29; // r14
  __m128i *v30; // rdi
  __int64 (__fastcall *v31)(__int64); // rax
  size_t v32; // rdi
  __int64 v33; // r8
  __int64 v34; // r9
  unsigned __int64 v35; // r15
  _QWORD *v36; // rax
  _QWORD *v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // rcx
  unsigned __int64 v40; // rdx
  char *v41; // rax
  __int64 v42; // rdx
  __m128i *p_src; // rdi
  size_t v44; // rsi
  __int64 v45; // rcx
  __int64 v46; // rdx
  size_t v47; // rdx
  __int128 v48; // [rsp+20h] [rbp-150h]
  __int64 v49; // [rsp+30h] [rbp-140h]
  __int128 v50; // [rsp+38h] [rbp-138h]
  const char *s2; // [rsp+40h] [rbp-130h]
  __int64 v52; // [rsp+50h] [rbp-120h]
  unsigned __int8 v55; // [rsp+7Fh] [rbp-F1h] BYREF
  void *v56[2]; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v57; // [rsp+90h] [rbp-E0h] BYREF
  __m128i v58; // [rsp+A0h] [rbp-D0h] BYREF
  __m128i v59; // [rsp+B0h] [rbp-C0h]
  _BYTE v60[16]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 (__fastcall *v61)(__int64 *); // [rsp+D0h] [rbp-A0h]
  __int64 v62; // [rsp+D8h] [rbp-98h]
  _BYTE v63[16]; // [rsp+E0h] [rbp-90h] BYREF
  __int64 (__fastcall *v64)(_QWORD *); // [rsp+F0h] [rbp-80h]
  __int64 v65; // [rsp+F8h] [rbp-78h]
  size_t n[2]; // [rsp+100h] [rbp-70h] BYREF
  __m128i src; // [rsp+110h] [rbp-60h] BYREF
  __int128 v68; // [rsp+120h] [rbp-50h]
  __int128 v69; // [rsp+130h] [rbp-40h]

  v6 = a3 + 32;
  v7 = a3 + 32 * a4;
  v10 = *(unsigned __int8 **)a3;
  if ( v7 != a3 )
  {
    while ( v7 != v6 )
    {
      v6 += 32;
      if ( *(unsigned __int8 **)(v6 - 32) != v10 )
        return 0;
    }
  }
  if ( *(_BYTE *)(a1 + 104) || (unsigned __int8)sub_C92250() )
    *(_BYTE *)(a3 + 25) = 1;
  v55 = 0;
  sub_26FC5B0(a1, (__int64)a5, v10, (unsigned __int8 **)&v55);
  v11 = v55;
  if ( !v55 )
    return 0;
  if ( (v10[32] & 0xFu) - 7 > 1 )
    goto LABEL_32;
  n[0] = (size_t)sub_BD5D20((__int64)v10);
  src.m128i_i64[0] = (__int64)".llvm.merged";
  LOWORD(v68) = 773;
  n[1] = v13;
  sub_CA0F50((__int64 *)v56, (void **)n);
  v52 = sub_B326A0((__int64)v10);
  if ( !v52 )
    goto LABEL_51;
  v14 = sub_BD5D20((__int64)v10);
  v16 = v15;
  s2 = v14;
  v17 = (const void *)sub_AA8810((_QWORD *)v52);
  if ( v18 != v16 || v18 && memcmp(v17, s2, v18) )
    goto LABEL_51;
  v49 = sub_BAA410(*(_QWORD *)a1, v56[0], (size_t)v56[1]);
  *(_DWORD *)(v49 + 8) = *(_DWORD *)(v52 + 8);
  sub_BA9600(n, *(_QWORD *)a1);
  v19 = _mm_loadu_si128(&src);
  v50 = v68;
  v58 = _mm_loadu_si128((const __m128i *)n);
  v48 = v69;
  v59 = v19;
  while ( *(_OWORD *)&v58 != v50 || *(_OWORD *)&v59 != v48 )
  {
    v20 = v60;
    v62 = 0;
    v21 = v60;
    v22 = &v58;
    v61 = sub_25AC5E0;
    v23 = sub_25AC5C0;
    if ( ((unsigned __int8)sub_25AC5C0 & 1) == 0 )
      goto LABEL_18;
    while ( 1 )
    {
      v23 = *(__int64 (__fastcall **)(__int64))((char *)v23 + v22->m128i_i64[0] - 1);
LABEL_18:
      v24 = v23((__int64)v22);
      if ( v24 )
        break;
      while ( 1 )
      {
        v20 += 16;
        if ( v63 == v20 )
LABEL_61:
          BUG();
        v27 = *((_QWORD *)v21 + 3);
        v23 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v21 + 2);
        v21 = v20;
        v22 = (__m128i *)((char *)&v58 + v27);
        if ( ((unsigned __int8)v23 & 1) != 0 )
          break;
        v24 = v23((__int64)v22);
        if ( v24 )
          goto LABEL_22;
      }
    }
LABEL_22:
    if ( v52 == *(_QWORD *)(v24 + 48) )
      sub_B2F990(v24, v49, v25, v26);
    v28 = (size_t *)v63;
    v65 = 0;
    v29 = (size_t *)v63;
    v30 = &v58;
    v64 = sub_25AC590;
    v31 = sub_25AC560;
    if ( ((unsigned __int8)sub_25AC560 & 1) != 0 )
LABEL_25:
      v31 = *(__int64 (__fastcall **)(__int64))((char *)v31 + v30->m128i_i64[0] - 1);
    while ( !(unsigned __int8)v31((__int64)v30) )
    {
      v28 += 2;
      if ( n == v28 )
        goto LABEL_61;
      v32 = v29[3];
      v31 = (__int64 (__fastcall *)(__int64))v29[2];
      v29 = v28;
      v30 = (__m128i *)((char *)&v58 + v32);
      if ( ((unsigned __int8)v31 & 1) != 0 )
        goto LABEL_25;
    }
  }
  v11 = (unsigned __int8)v11;
LABEL_51:
  *((_WORD *)v10 + 16) = *((_WORD *)v10 + 16) & 0xBFC0 | 0x4010;
  LOWORD(v68) = 260;
  n[0] = (size_t)v56;
  sub_BD6B50(v10, (const char **)n);
  if ( v56[0] != &v57 )
    j_j___libc_free_0((unsigned __int64)v56[0]);
LABEL_32:
  sub_B2F930(n, (__int64)v10);
  v35 = sub_B2F650(n[0], n[1]);
  if ( (__m128i *)n[0] != &src )
    j_j___libc_free_0(n[0]);
  v36 = *(_QWORD **)(a2 + 16);
  v37 = (_QWORD *)(a2 + 8);
  if ( v36 )
  {
    do
    {
      while ( 1 )
      {
        v38 = v36[2];
        v39 = v36[3];
        if ( v35 <= v36[4] )
          break;
        v36 = (_QWORD *)v36[3];
        if ( !v39 )
          goto LABEL_39;
      }
      v37 = v36;
      v36 = (_QWORD *)v36[2];
    }
    while ( v38 );
LABEL_39:
    if ( v37 != (_QWORD *)(a2 + 8) && v35 >= v37[4] )
    {
      v40 = *(unsigned __int8 *)(a2 + 343) | (unsigned __int64)(v37 + 4) & 0xFFFFFFFFFFFFFFF8LL;
      n[0] = v40;
      if ( (v40 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        sub_26F7CF0(a5, (__int64 *)n, v40, v39, v33, v34);
    }
  }
  *(_DWORD *)a6 = 1;
  v41 = (char *)sub_BD5D20((__int64)v10);
  n[0] = (size_t)&src;
  sub_26F6410((__int64 *)n, v41, (__int64)&v41[v42]);
  p_src = *(__m128i **)(a6 + 8);
  if ( (__m128i *)n[0] != &src )
  {
    v44 = n[1];
    v45 = src.m128i_i64[0];
    if ( p_src == (__m128i *)(a6 + 24) )
    {
      *(_QWORD *)(a6 + 8) = n[0];
      *(_QWORD *)(a6 + 16) = v44;
      *(_QWORD *)(a6 + 24) = v45;
    }
    else
    {
      v46 = *(_QWORD *)(a6 + 24);
      *(_QWORD *)(a6 + 8) = n[0];
      *(_QWORD *)(a6 + 16) = v44;
      *(_QWORD *)(a6 + 24) = v45;
      if ( p_src )
      {
        n[0] = (size_t)p_src;
        src.m128i_i64[0] = v46;
        goto LABEL_47;
      }
    }
    n[0] = (size_t)&src;
    p_src = &src;
    goto LABEL_47;
  }
  v47 = n[1];
  if ( n[1] )
  {
    if ( n[1] == 1 )
      p_src->m128i_i8[0] = src.m128i_i8[0];
    else
      memcpy(p_src, &src, n[1]);
    v47 = n[1];
    p_src = *(__m128i **)(a6 + 8);
  }
  *(_QWORD *)(a6 + 16) = v47;
  p_src->m128i_i8[v47] = 0;
  p_src = (__m128i *)n[0];
LABEL_47:
  n[1] = 0;
  p_src->m128i_i8[0] = 0;
  if ( (__m128i *)n[0] != &src )
    j_j___libc_free_0(n[0]);
  return v11;
}
