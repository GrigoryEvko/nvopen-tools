// Function: sub_1954210
// Address: 0x1954210
//
__int64 __fastcall sub_1954210(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r13
  unsigned int v12; // ebx
  __int64 v13; // rdx
  _QWORD *v14; // rcx
  _QWORD *v15; // rax
  signed __int64 v16; // rdx
  _QWORD *v17; // rsi
  __int64 **v18; // r15
  _QWORD *v19; // r14
  unsigned __int8 v20; // al
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 *v23; // rax
  __int64 v25; // rax
  __int64 v26; // r13
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r11
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  double v36; // xmm4_8
  double v37; // xmm5_8
  unsigned __int64 v38; // rax
  int v39; // eax
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // r14
  __m128i *v42; // rsi
  __int64 v43; // rax
  __m128i *v44; // rsi
  __int64 v45; // rcx
  __m128i *v46; // rax
  unsigned __int64 v47; // rcx
  __int64 v48; // [rsp+0h] [rbp-80h]
  __int64 v49; // [rsp+0h] [rbp-80h]
  __int64 v50; // [rsp+8h] [rbp-78h]
  __int64 v51; // [rsp+8h] [rbp-78h]
  __int64 **v53; // [rsp+18h] [rbp-68h]
  __int64 v54; // [rsp+18h] [rbp-68h]
  int v55; // [rsp+18h] [rbp-68h]
  __m128i v56; // [rsp+20h] [rbp-60h] BYREF
  const __m128i *v57; // [rsp+30h] [rbp-50h] BYREF
  __m128i *v58; // [rsp+38h] [rbp-48h]
  const __m128i *v59; // [rsp+40h] [rbp-40h]

  if ( sub_1377F70(a1 + 56, a2) )
    return 0;
  v11 = *(_QWORD *)(a2 + 48);
  v12 = 0;
  while ( 1 )
  {
    if ( !v11 )
      BUG();
    if ( *(_BYTE *)(v11 - 8) != 77 )
      return 0;
    v13 = 24LL * (*(_DWORD *)(v11 - 4) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v11 - 1) & 0x40) != 0 )
    {
      v15 = *(_QWORD **)(v11 - 32);
      v14 = &v15[(unsigned __int64)v13 / 8];
    }
    else
    {
      v14 = (_QWORD *)(v11 - 24);
      v15 = (_QWORD *)(v11 - 24 - v13);
    }
    v16 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
    if ( v16 >> 2 )
    {
      v17 = &v15[12 * (v16 >> 2)];
      while ( 1 )
      {
        if ( *(_BYTE *)(*v15 + 16LL) == 13 )
          goto LABEL_14;
        if ( *(_BYTE *)(v15[3] + 16LL) == 13 )
        {
          v15 += 3;
          goto LABEL_14;
        }
        if ( *(_BYTE *)(v15[6] + 16LL) == 13 )
        {
          v15 += 6;
          goto LABEL_14;
        }
        if ( *(_BYTE *)(v15[9] + 16LL) == 13 )
          break;
        v15 += 12;
        if ( v17 == v15 )
        {
          v16 = 0xAAAAAAAAAAAAAAABLL * (v14 - v15);
          goto LABEL_51;
        }
      }
      v15 += 9;
      goto LABEL_14;
    }
LABEL_51:
    if ( v16 == 2 )
      goto LABEL_60;
    if ( v16 == 3 )
    {
      if ( *(_BYTE *)(*v15 + 16LL) == 13 )
        goto LABEL_14;
      v15 += 3;
LABEL_60:
      if ( *(_BYTE *)(*v15 + 16LL) == 13 )
        goto LABEL_14;
      v15 += 3;
      goto LABEL_62;
    }
    if ( v16 != 1 )
      goto LABEL_54;
LABEL_62:
    if ( *(_BYTE *)(*v15 + 16LL) != 13 )
      goto LABEL_54;
LABEL_14:
    if ( v15 != v14 )
    {
      v18 = *(__int64 ***)(v11 - 16);
      if ( v18 )
        break;
    }
LABEL_54:
    v11 = *(_QWORD *)(v11 + 8);
  }
  while ( 1 )
  {
    v19 = sub_1648700((__int64)v18);
    v20 = *((_BYTE *)v19 + 16);
    if ( v20 > 0x17u )
    {
      if ( v20 == 75 )
      {
        if ( a2 != v19[5] )
          goto LABEL_18;
        v21 = v19[1];
        if ( !v21 )
          goto LABEL_18;
        if ( *(_QWORD *)(v21 + 8) )
          goto LABEL_18;
        if ( *(_BYTE *)(v19[3 * (1 - (unsigned int)sub_1648720((__int64)v18)) - 6] + 16LL) != 13 )
          goto LABEL_18;
        v53 = (__int64 **)v19[1];
        v22 = sub_1648700((__int64)v53);
        v19 = v22;
        if ( *((_BYTE *)v22 + 16) != 79 )
          goto LABEL_18;
        if ( a2 != v22[5] )
          goto LABEL_18;
        v23 = (__int64 *)*(v22 - 9);
        if ( *v53 != v23 )
          goto LABEL_18;
      }
      else
      {
        if ( v20 != 79 )
          goto LABEL_18;
        if ( a2 != v19[5] )
          goto LABEL_18;
        v23 = (__int64 *)*(v19 - 9);
        if ( *v18 != v23 )
          goto LABEL_18;
      }
      if ( v23 && sub_1642F90(*v23, 1) )
        break;
    }
LABEL_18:
    v18 = (__int64 **)v18[1];
    if ( !v18 )
      goto LABEL_54;
  }
  v25 = sub_1AA92B0(*(v19 - 9), v19, 0, 0, 0, 0);
  v26 = v19[5];
  v48 = v25;
  v54 = *(_QWORD *)(v25 + 40);
  LOWORD(v59) = 257;
  v27 = *v19;
  v28 = sub_1648B60(64);
  v32 = v28;
  if ( v28 )
  {
    v50 = v28;
    sub_15F1EA0(v28, v27, 53, 0, 0, (__int64)v19);
    *(_DWORD *)(v50 + 56) = 2;
    sub_164B780(v50, (__int64 *)&v57);
    sub_1648880(v50, *(_DWORD *)(v50 + 56), 1);
    v32 = v50;
  }
  v51 = v32;
  sub_1704F80(v32, *(v19 - 6), *(_QWORD *)(v48 + 40), v29, v30, v31);
  sub_1704F80(v51, *(v19 - 3), a2, v33, v34, v35);
  sub_164D160((__int64)v19, v51, a3, a4, a5, a6, v36, v37, a9, a10);
  sub_15F20C0(v19);
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v38 = sub_157EBA0(v26);
  v39 = sub_15F4D60(v38);
  sub_1953AE0(&v57, (unsigned int)(2 * v39 + 3));
  v56.m128i_i64[0] = a2;
  v56.m128i_i64[1] = v26 & 0xFFFFFFFFFFFFFFFBLL;
  sub_19541D0((__int64)&v57, &v56);
  v56.m128i_i64[0] = a2;
  v56.m128i_i64[1] = v54 & 0xFFFFFFFFFFFFFFFBLL;
  sub_19541D0((__int64)&v57, &v56);
  v56.m128i_i64[1] = v26 & 0xFFFFFFFFFFFFFFFBLL;
  v56.m128i_i64[0] = v54;
  sub_19541D0((__int64)&v57, &v56);
  v40 = sub_157EBA0(v26);
  if ( v40 )
  {
    v55 = sub_15F4D60(v40);
    v41 = sub_157EBA0(v26);
    if ( v55 )
    {
      do
      {
        while ( 1 )
        {
          v43 = sub_15F4DF0(v41, v12);
          v44 = v58;
          v56.m128i_i64[0] = a2;
          v45 = v43;
          v56.m128i_i64[1] = v43 | 4;
          v46 = (__m128i *)v59;
          v47 = v45 & 0xFFFFFFFFFFFFFFFBLL;
          if ( v58 == v59 )
            break;
          if ( v58 )
          {
            *v58 = _mm_loadu_si128(&v56);
            v44 = v58;
            v46 = (__m128i *)v59;
          }
          v42 = v44 + 1;
          v56.m128i_i64[0] = v26;
          v58 = v42;
          v56.m128i_i64[1] = v47;
          if ( v46 == v42 )
            goto LABEL_46;
LABEL_40:
          *v42 = _mm_loadu_si128(&v56);
          v42 = v58;
LABEL_41:
          ++v12;
          v58 = v42 + 1;
          if ( v55 == v12 )
            goto LABEL_47;
        }
        v49 = v47;
        sub_17F2860(&v57, v58, &v56);
        v56.m128i_i64[0] = v26;
        v42 = v58;
        v56.m128i_i64[1] = v49;
        if ( v59 != v58 )
        {
          if ( !v58 )
            goto LABEL_41;
          goto LABEL_40;
        }
LABEL_46:
        ++v12;
        sub_17F2860(&v57, v42, &v56);
      }
      while ( v55 != v12 );
    }
  }
LABEL_47:
  sub_15CD9D0(*(_QWORD *)(a1 + 24), v57->m128i_i64, v58 - v57);
  if ( v57 )
    j_j___libc_free_0(v57, (char *)v59 - (char *)v57);
  return 1;
}
