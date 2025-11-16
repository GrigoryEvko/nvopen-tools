// Function: sub_1233B90
// Address: 0x1233b90
//
__int64 __fastcall sub_1233B90(__int64 a1, __int64 *a2, __int64 *a3)
{
  unsigned __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // r13d
  __int64 v9; // rdi
  __int64 v10; // rax
  __m128i *v11; // rax
  __int64 v12; // rcx
  __m128i *v13; // rax
  __int64 v14; // rcx
  __m128i *v15; // r10
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rcx
  __m128i *v19; // rax
  __m128i *v20; // rcx
  __m128i *v21; // rdx
  __int64 v22; // rcx
  __m128i *v23; // rax
  __int64 v24; // rcx
  const char *v25; // rax
  unsigned int *v26; // rbx
  __int64 v27; // r15
  _QWORD *v28; // rax
  __int64 v29; // r14
  _BOOL4 v30; // r13d
  unsigned __int64 v31; // [rsp+8h] [rbp-178h]
  __int64 v32; // [rsp+8h] [rbp-178h]
  unsigned __int64 v34; // [rsp+18h] [rbp-168h]
  __int64 v35; // [rsp+18h] [rbp-168h]
  char v36; // [rsp+2Fh] [rbp-151h] BYREF
  __int64 v37; // [rsp+30h] [rbp-150h] BYREF
  __int64 v38; // [rsp+38h] [rbp-148h] BYREF
  unsigned int *v39; // [rsp+40h] [rbp-140h] BYREF
  __int64 v40; // [rsp+48h] [rbp-138h]
  _BYTE v41[16]; // [rsp+50h] [rbp-130h] BYREF
  __int64 v42[2]; // [rsp+60h] [rbp-120h] BYREF
  __int64 v43; // [rsp+70h] [rbp-110h] BYREF
  __m128i *v44; // [rsp+80h] [rbp-100h] BYREF
  __int64 v45; // [rsp+88h] [rbp-F8h]
  __m128i v46; // [rsp+90h] [rbp-F0h] BYREF
  __m128i *v47; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v48; // [rsp+A8h] [rbp-D8h]
  __m128i v49; // [rsp+B0h] [rbp-D0h] BYREF
  _QWORD *v50; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v51; // [rsp+C8h] [rbp-B8h]
  _QWORD v52[2]; // [rsp+D0h] [rbp-B0h] BYREF
  __m128i *v53; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v54; // [rsp+E8h] [rbp-98h]
  __m128i v55; // [rsp+F0h] [rbp-90h] BYREF
  _QWORD v56[2]; // [rsp+100h] [rbp-80h] BYREF
  __m128i v57; // [rsp+110h] [rbp-70h] BYREF
  _QWORD v58[4]; // [rsp+120h] [rbp-60h] BYREF
  __int16 v59; // [rsp+140h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 232);
  v6 = (__int64)&v37;
  v39 = (unsigned int *)v41;
  v40 = 0x400000000LL;
  v34 = v5;
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v37, a3) )
    goto LABEL_2;
  v6 = 4;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected comma after insertvalue operand") )
    goto LABEL_2;
  v6 = (__int64)&v38;
  v31 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v38, a3) )
    goto LABEL_2;
  v6 = (__int64)&v39;
  if ( (unsigned __int8)sub_120E620(a1, (__int64)&v39, &v36) )
    goto LABEL_2;
  v9 = *(_QWORD *)(v37 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 15 > 1 )
  {
    HIBYTE(v59) = 1;
    v25 = "insertvalue operand must be aggregate type";
LABEL_41:
    v6 = v34;
    v58[0] = v25;
    LOBYTE(v59) = 3;
    sub_11FD800(a1 + 176, v34, (__int64)v58, 1);
LABEL_2:
    v7 = 1;
    goto LABEL_3;
  }
  v10 = sub_B501B0(v9, v39, (unsigned int)v40);
  if ( !v10 )
  {
    HIBYTE(v59) = 1;
    v25 = "invalid indices for insertvalue";
    goto LABEL_41;
  }
  if ( v10 != *(_QWORD *)(v38 + 8) )
  {
    sub_1207630((__int64 *)&v50, v10);
    sub_1207630(v42, *(_QWORD *)(v38 + 8));
    v11 = (__m128i *)sub_2241130(v42, 0, 0, "insertvalue operand and field disagree in type: '", 49);
    v44 = &v46;
    if ( (__m128i *)v11->m128i_i64[0] == &v11[1] )
    {
      v46 = _mm_loadu_si128(v11 + 1);
    }
    else
    {
      v44 = (__m128i *)v11->m128i_i64[0];
      v46.m128i_i64[0] = v11[1].m128i_i64[0];
    }
    v12 = v11->m128i_i64[1];
    v11[1].m128i_i8[0] = 0;
    v45 = v12;
    v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
    v11->m128i_i64[1] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v45) <= 0xD )
      goto LABEL_52;
    v13 = (__m128i *)sub_2241490(&v44, "' instead of '", 14, v12);
    v47 = &v49;
    if ( (__m128i *)v13->m128i_i64[0] == &v13[1] )
    {
      v49 = _mm_loadu_si128(v13 + 1);
    }
    else
    {
      v47 = (__m128i *)v13->m128i_i64[0];
      v49.m128i_i64[0] = v13[1].m128i_i64[0];
    }
    v14 = v13->m128i_i64[1];
    v13[1].m128i_i8[0] = 0;
    v48 = v14;
    v13->m128i_i64[0] = (__int64)v13[1].m128i_i64;
    v15 = v47;
    v13->m128i_i64[1] = 0;
    v16 = 15;
    v17 = 15;
    if ( v15 != &v49 )
      v17 = v49.m128i_i64[0];
    v18 = v48 + v51;
    if ( v48 + v51 <= v17 )
      goto LABEL_23;
    if ( v50 != v52 )
      v16 = v52[0];
    if ( v18 <= v16 )
    {
      v19 = (__m128i *)sub_2241130(&v50, 0, 0, v15, v48);
      v53 = &v55;
      v20 = (__m128i *)v19->m128i_i64[0];
      v21 = v19 + 1;
      if ( (__m128i *)v19->m128i_i64[0] != &v19[1] )
        goto LABEL_24;
    }
    else
    {
LABEL_23:
      v19 = (__m128i *)sub_2241490(&v47, v50, v51, v18);
      v53 = &v55;
      v20 = (__m128i *)v19->m128i_i64[0];
      v21 = v19 + 1;
      if ( (__m128i *)v19->m128i_i64[0] != &v19[1] )
      {
LABEL_24:
        v53 = v20;
        v55.m128i_i64[0] = v19[1].m128i_i64[0];
        goto LABEL_25;
      }
    }
    v55 = _mm_loadu_si128(v19 + 1);
LABEL_25:
    v22 = v19->m128i_i64[1];
    v54 = v22;
    v19->m128i_i64[0] = (__int64)v21;
    v19->m128i_i64[1] = 0;
    v19[1].m128i_i8[0] = 0;
    if ( v54 != 0x3FFFFFFFFFFFFFFFLL )
    {
      v23 = (__m128i *)sub_2241490(&v53, "'", 1, v22);
      v56[0] = &v57;
      if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
      {
        v57 = _mm_loadu_si128(v23 + 1);
      }
      else
      {
        v56[0] = v23->m128i_i64[0];
        v57.m128i_i64[0] = v23[1].m128i_i64[0];
      }
      v24 = v23->m128i_i64[1];
      v23[1].m128i_i8[0] = 0;
      v6 = v31;
      v56[1] = v24;
      v23->m128i_i64[0] = (__int64)v23[1].m128i_i64;
      v23->m128i_i64[1] = 0;
      v59 = 260;
      v58[0] = v56;
      sub_11FD800(a1 + 176, v31, (__int64)v58, 1);
      if ( (__m128i *)v56[0] != &v57 )
      {
        v6 = v57.m128i_i64[0] + 1;
        j_j___libc_free_0(v56[0], v57.m128i_i64[0] + 1);
      }
      if ( v53 != &v55 )
      {
        v6 = v55.m128i_i64[0] + 1;
        j_j___libc_free_0(v53, v55.m128i_i64[0] + 1);
      }
      if ( v47 != &v49 )
      {
        v6 = v49.m128i_i64[0] + 1;
        j_j___libc_free_0(v47, v49.m128i_i64[0] + 1);
      }
      if ( v44 != &v46 )
      {
        v6 = v46.m128i_i64[0] + 1;
        j_j___libc_free_0(v44, v46.m128i_i64[0] + 1);
      }
      if ( (__int64 *)v42[0] != &v43 )
      {
        v6 = v43 + 1;
        j_j___libc_free_0(v42[0], v43 + 1);
      }
      if ( v50 != v52 )
      {
        v6 = v52[0] + 1LL;
        j_j___libc_free_0(v50, v52[0] + 1LL);
      }
      goto LABEL_2;
    }
LABEL_52:
    sub_4262D8((__int64)"basic_string::append");
  }
  v32 = v38;
  v26 = v39;
  v59 = 257;
  v27 = v37;
  v35 = (unsigned int)v40;
  v6 = unk_3F148BC;
  v28 = sub_BD2C40(104, unk_3F148BC);
  v29 = (__int64)v28;
  if ( v28 )
  {
    sub_B44260((__int64)v28, *(_QWORD *)(v27 + 8), 65, 2u, 0, 0);
    *(_QWORD *)(v29 + 80) = 0x400000000LL;
    *(_QWORD *)(v29 + 72) = v29 + 88;
    v6 = v27;
    sub_B4FD20(v29, v27, v32, v26, v35, (__int64)v58);
  }
  v30 = v36 != 0;
  *a2 = v29;
  v7 = 2 * v30;
LABEL_3:
  if ( v39 != (unsigned int *)v41 )
    _libc_free(v39, v6);
  return v7;
}
