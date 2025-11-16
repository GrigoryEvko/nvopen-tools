// Function: sub_107B020
// Address: 0x107b020
//
void __fastcall sub_107B020(__int64 a1, unsigned int a2, _BYTE *a3, __int64 a4, __int64 a5)
{
  char *v6; // r9
  __m128i *v7; // r8
  __int64 v10; // rcx
  __int64 v11; // r15
  __m128i *v12; // rax
  __m128i *v13; // r14
  __m128i *v14; // rdi
  __m128i *v15; // rax
  __m128i v16; // xmm0
  __m128i v17; // xmm1
  __int64 v18; // rsi
  const __m128i *v19; // rax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rcx
  __m128i *v23; // rax
  __int64 v24; // rcx
  __m128i *v25; // rdx
  __m128i *v26; // rax
  __m128i *v27; // rbx
  unsigned __int64 v28; // r15
  unsigned int v29; // eax
  __int32 v30; // esi
  unsigned __int64 v31; // r14
  __int64 v32; // rdi
  _BYTE *v33; // rax
  __int64 v34; // r13
  char v35; // si
  char v36; // al
  char *v37; // rax
  __int64 v38; // r13
  char v39; // si
  char v40; // al
  char *v41; // rax
  char v42; // r13
  __int64 v43; // r14
  __int64 v44; // r15
  char v45; // cl
  char v46; // si
  char *v47; // rax
  char v48; // al
  __m128i *v49; // [rsp+0h] [rbp-E0h]
  __m128i *src; // [rsp+8h] [rbp-D8h]
  __int64 v51; // [rsp+10h] [rbp-D0h]
  __int64 v52; // [rsp+18h] [rbp-C8h]
  __m128i *v54; // [rsp+20h] [rbp-C0h]
  char v56; // [rsp+2Ch] [rbp-B4h]
  __int64 v57[4]; // [rsp+30h] [rbp-B0h] BYREF
  _QWORD *v58; // [rsp+50h] [rbp-90h] BYREF
  __int64 v59; // [rsp+58h] [rbp-88h]
  _QWORD v60[2]; // [rsp+60h] [rbp-80h] BYREF
  _QWORD *v61; // [rsp+70h] [rbp-70h] BYREF
  __int64 v62; // [rsp+78h] [rbp-68h]
  _QWORD v63[2]; // [rsp+80h] [rbp-60h] BYREF
  _OWORD *v64; // [rsp+90h] [rbp-50h]
  size_t v65; // [rsp+98h] [rbp-48h]
  _OWORD v66[4]; // [rsp+A0h] [rbp-40h] BYREF

  v6 = *(char **)(a5 + 8);
  v7 = *(__m128i **)a5;
  if ( v7 == (__m128i *)v6 )
    return;
  v10 = 0xCCCCCCCCCCCCCCCDLL * ((v6 - (char *)v7) >> 3);
  if ( v6 - (char *)v7 <= 0 )
  {
LABEL_57:
    v11 = 0;
    v13 = 0;
    sub_10785D0(v7, v6);
  }
  else
  {
    while ( 1 )
    {
      v49 = (__m128i *)v6;
      src = v7;
      v51 = v10;
      v11 = 40 * v10;
      v52 = 40 * v10;
      v12 = (__m128i *)sub_2207800(40 * v10, &unk_435FF63);
      v7 = src;
      v6 = (char *)v49;
      v13 = v12;
      if ( v12 )
        break;
      v10 = v51 >> 1;
      if ( !(v51 >> 1) )
        goto LABEL_57;
    }
    v14 = (__m128i *)((char *)v12 + v11);
    *v12 = _mm_loadu_si128(src);
    v12[1] = _mm_loadu_si128(src + 1);
    v12[2].m128i_i64[0] = src[2].m128i_i64[0];
    v15 = (__m128i *)((char *)v12 + 40);
    if ( v14 == (__m128i *)&v13[2].m128i_u64[1] )
    {
      v19 = v13;
    }
    else
    {
      do
      {
        v16 = _mm_loadu_si128((__m128i *)((char *)v15 - 40));
        v17 = _mm_loadu_si128((__m128i *)((char *)v15 - 24));
        v15 = (__m128i *)((char *)v15 + 40);
        v18 = v15[-3].m128i_i64[0];
        *(__m128i *)((char *)v15 - 40) = v16;
        *(__m128i *)((char *)v15 - 24) = v17;
        v15[-1].m128i_i64[1] = v18;
      }
      while ( v14 != v15 );
      v19 = (__m128i *)((char *)v13 + v52 - 40);
    }
    *src = _mm_loadu_si128(v19);
    src[1] = _mm_loadu_si128(v19 + 1);
    src[2].m128i_i64[0] = v19[2].m128i_i64[0];
    sub_1079160(src, v49, v13, (const __m128i *)v51);
  }
  j_j___libc_free_0(v13, v11);
  if ( a3 )
  {
    v61 = v63;
    sub_10772D0((__int64 *)&v61, a3, (__int64)&a3[a4]);
  }
  else
  {
    v62 = 0;
    v61 = v63;
    LOBYTE(v63[0]) = 0;
  }
  v58 = v60;
  sub_10772D0((__int64 *)&v58, "reloc.", (__int64)"");
  v20 = 15;
  v21 = 15;
  if ( v58 != v60 )
    v21 = v60[0];
  v22 = v59 + v62;
  if ( v59 + v62 > v21 )
  {
    if ( v61 != v63 )
      v20 = v63[0];
    if ( v22 <= v20 )
    {
      v23 = (__m128i *)sub_2241130(&v61, 0, 0, v58, v59);
      v64 = v66;
      v24 = v23->m128i_i64[0];
      v25 = v23 + 1;
      if ( (__m128i *)v23->m128i_i64[0] != &v23[1] )
        goto LABEL_17;
LABEL_60:
      v66[0] = _mm_loadu_si128(v23 + 1);
      goto LABEL_18;
    }
  }
  v23 = (__m128i *)sub_2241490(&v58, v61, v62, v22);
  v64 = v66;
  v24 = v23->m128i_i64[0];
  v25 = v23 + 1;
  if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
    goto LABEL_60;
LABEL_17:
  v64 = (_OWORD *)v24;
  *(_QWORD *)&v66[0] = v23[1].m128i_i64[0];
LABEL_18:
  v65 = v23->m128i_u64[1];
  v23->m128i_i64[0] = (__int64)v25;
  v23->m128i_i64[1] = 0;
  v23[1].m128i_i8[0] = 0;
  sub_1079790(a1, (__int64)v57, (__int64)v64, v65);
  if ( v64 != v66 )
    j_j___libc_free_0(v64, *(_QWORD *)&v66[0] + 1LL);
  if ( v58 != v60 )
    j_j___libc_free_0(v58, v60[0] + 1LL);
  if ( v61 != v63 )
    j_j___libc_free_0(v61, v63[0] + 1LL);
  sub_107A5C0(a2, **(_QWORD **)(a1 + 104), 0);
  sub_107A5C0(0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(a5 + 8) - *(_QWORD *)a5) >> 3), **(_QWORD **)(a1 + 104), 0);
  v26 = *(__m128i **)(a5 + 8);
  v27 = *(__m128i **)a5;
  v54 = v26;
  if ( v27 != v26 )
  {
    while ( 1 )
    {
      v28 = *(_QWORD *)(v27[2].m128i_i64[0] + 160) + v27->m128i_i64[0];
      v29 = sub_1078680(a1, v27->m128i_i64[1], v27[1].m128i_i32[2]);
      v30 = v27[1].m128i_i32[2];
      v31 = v29;
      v32 = **(_QWORD **)(a1 + 104);
      v33 = *(_BYTE **)(v32 + 32);
      if ( (unsigned __int64)v33 >= *(_QWORD *)(v32 + 24) )
      {
        sub_CB5D20(v32, v30);
      }
      else
      {
        *(_QWORD *)(v32 + 32) = v33 + 1;
        *v33 = v30;
      }
      v34 = **(_QWORD **)(a1 + 104);
      do
      {
        while ( 1 )
        {
          v35 = v28 & 0x7F;
          v36 = v28 & 0x7F | 0x80;
          v28 >>= 7;
          if ( v28 )
            v35 = v36;
          v37 = *(char **)(v34 + 32);
          if ( (unsigned __int64)v37 >= *(_QWORD *)(v34 + 24) )
            break;
          *(_QWORD *)(v34 + 32) = v37 + 1;
          *v37 = v35;
          if ( !v28 )
            goto LABEL_33;
        }
        sub_CB5D20(v34, v35);
      }
      while ( v28 );
LABEL_33:
      v38 = **(_QWORD **)(a1 + 104);
      do
      {
        while ( 1 )
        {
          v39 = v31 & 0x7F;
          v40 = v31 & 0x7F | 0x80;
          v31 >>= 7;
          if ( v31 )
            v39 = v40;
          v41 = *(char **)(v38 + 32);
          if ( (unsigned __int64)v41 >= *(_QWORD *)(v38 + 24) )
            break;
          *(_QWORD *)(v38 + 32) = v41 + 1;
          *v41 = v39;
          if ( !v31 )
            goto LABEL_39;
        }
        sub_CB5D20(v38, v39);
      }
      while ( v31 );
LABEL_39:
      v42 = sub_1249500(v27[1].m128i_u32[2]);
      if ( v42 )
        break;
LABEL_40:
      v27 = (__m128i *)((char *)v27 + 40);
      if ( v54 == v27 )
        goto LABEL_41;
    }
    v43 = v27[1].m128i_i64[0];
    v44 = **(_QWORD **)(a1 + 104);
    while ( 1 )
    {
      v48 = v43;
      v46 = v43 & 0x7F;
      v43 >>= 7;
      if ( !v43 )
        break;
      if ( v43 != -1 )
        goto LABEL_45;
      v45 = 0;
      if ( (v48 & 0x40) == 0 )
        goto LABEL_45;
      v47 = *(char **)(v44 + 32);
      if ( (unsigned __int64)v47 >= *(_QWORD *)(v44 + 24) )
      {
LABEL_53:
        v56 = v45;
        sub_CB5D20(v44, v46);
        v45 = v56;
        goto LABEL_48;
      }
LABEL_47:
      *(_QWORD *)(v44 + 32) = v47 + 1;
      *v47 = v46;
LABEL_48:
      if ( !v45 )
        goto LABEL_40;
    }
    v45 = 0;
    if ( (v48 & 0x40) != 0 )
    {
LABEL_45:
      v46 |= 0x80u;
      v45 = v42;
    }
    v47 = *(char **)(v44 + 32);
    if ( (unsigned __int64)v47 >= *(_QWORD *)(v44 + 24) )
      goto LABEL_53;
    goto LABEL_47;
  }
LABEL_41:
  sub_1077B30(a1, v57);
}
