// Function: sub_336EA20
// Address: 0x336ea20
//
__int64 *__fastcall sub_336EA20(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 ***v6; // rax
  __int64 v7; // r15
  __int64 v8; // rax
  __int32 v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r9
  __int64 v12; // r8
  unsigned __int16 v13; // cx
  _QWORD *v14; // rax
  unsigned __int16 *v15; // rax
  __int32 v16; // esi
  __int64 v17; // r15
  __int64 v18; // r15
  __m128i v19; // xmm0
  __int64 v20; // rax
  __int64 v21; // rbx
  unsigned __int64 v22; // r13
  __int64 v23; // rax
  __int64 *result; // rax
  __int64 v25; // rax
  __int64 v26; // r10
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rdx
  char v31; // al
  __int64 v32; // rax
  __int64 v33; // r14
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // rsi
  __int32 v37; // ecx
  int v38; // esi
  __int32 v39; // r8d
  unsigned int v40; // edx
  bool v41; // al
  __int64 v42; // rax
  int v43; // edx
  __int64 v44; // rax
  __int64 v45; // rsi
  bool v46; // al
  __int64 v47; // rdi
  char v48; // [rsp+Bh] [rbp-95h]
  unsigned __int16 v49; // [rsp+Ch] [rbp-94h]
  unsigned __int16 v50; // [rsp+10h] [rbp-90h]
  __int64 v51; // [rsp+10h] [rbp-90h]
  __int64 *v52; // [rsp+18h] [rbp-88h]
  __int64 v53; // [rsp+18h] [rbp-88h]
  __int64 v54; // [rsp+18h] [rbp-88h]
  __int64 v55; // [rsp+18h] [rbp-88h]
  __int64 v56; // [rsp+18h] [rbp-88h]
  __m128i v57; // [rsp+20h] [rbp-80h] BYREF
  __m128i v58; // [rsp+30h] [rbp-70h] BYREF
  __int64 v59; // [rsp+40h] [rbp-60h] BYREF
  __int64 v60; // [rsp+48h] [rbp-58h]
  __int64 v61; // [rsp+50h] [rbp-50h] BYREF
  __int64 v62; // [rsp+58h] [rbp-48h]
  __int64 v63; // [rsp+60h] [rbp-40h]
  __int64 v64; // [rsp+68h] [rbp-38h]

  v3 = a2;
  v6 = (__int64 ***)a1[2];
  v7 = *a1;
  v57.m128i_i32[0] = a3;
  v52 = **v6;
  v8 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1[1] + 864) + 40LL));
  v9 = sub_2D5BAE0(v7, v8, v52, 0);
  v12 = v57.m128i_u32[0];
  v58.m128i_i32[0] = v9;
  v13 = v9;
  v14 = (_QWORD *)a1[2];
  v58.m128i_i64[1] = v10;
  *v14 += 8LL;
  v15 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16 * v12);
  v16 = *v15;
  v17 = *((_QWORD *)v15 + 1);
  if ( v13 == (_WORD)v16 )
  {
    if ( v13 )
      goto LABEL_3;
    if ( v10 == v17 )
      goto LABEL_28;
    v62 = *((_QWORD *)v15 + 1);
    LOWORD(v61) = 0;
LABEL_9:
    v50 = v13;
    v53 = v12;
    v57.m128i_i32[0] = v16;
    v25 = sub_3007260((__int64)&v61);
    v13 = v50;
    v12 = v53;
    v63 = v25;
    v26 = v25;
    v64 = v27;
    v48 = v27;
    if ( !v50 )
      goto LABEL_10;
    goto LABEL_34;
  }
  LOWORD(v61) = *v15;
  v62 = v17;
  if ( !(_WORD)v16 )
    goto LABEL_9;
  if ( (_WORD)v16 == 1 || (unsigned __int16)(v16 - 504) <= 7u )
    goto LABEL_47;
  v26 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v16 - 16];
  v48 = byte_444C4A0[16 * (unsigned __int16)v16 - 8];
  if ( v13 )
  {
LABEL_34:
    if ( v13 != 1 && (unsigned __int16)(v13 - 504) > 7u )
    {
      v31 = byte_444C4A0[16 * v13 - 8];
      if ( *(_QWORD *)&byte_444C4A0[16 * v13 - 16] != v26 )
        goto LABEL_11;
      goto LABEL_37;
    }
LABEL_47:
    BUG();
  }
LABEL_10:
  v49 = v13;
  v51 = v12;
  v54 = v26;
  v57.m128i_i32[0] = v16;
  v28 = sub_3007260((__int64)&v58);
  v13 = v49;
  v29 = v28;
  v31 = v30;
  v12 = v51;
  v62 = v30;
  v61 = v29;
  if ( v29 != v54 )
  {
LABEL_11:
    if ( v13 != (_WORD)v16 )
    {
      if ( v13 )
      {
        if ( (unsigned __int16)(v13 - 2) > 7u
          && (unsigned __int16)(v13 - 17) > 0x6Cu
          && (unsigned __int16)(v13 - 176) > 0x1Fu )
        {
          goto LABEL_3;
        }
        goto LABEL_16;
      }
LABEL_29:
      v55 = v12;
      v57.m128i_i32[0] = v16;
      v41 = sub_3007070((__int64)&v58);
      v12 = v55;
      if ( !v41 )
        goto LABEL_3;
LABEL_16:
      LOWORD(v59) = v16;
      v60 = v17;
      if ( (_WORD)v16 )
      {
        if ( (unsigned __int16)(v16 - 2) > 7u
          && (unsigned __int16)(v16 - 17) > 0x6Cu
          && (unsigned __int16)(v16 - 176) > 0x1Fu )
        {
          goto LABEL_3;
        }
      }
      else
      {
        v57.m128i_i64[0] = v12;
        v46 = sub_3007070((__int64)&v59);
        v12 = v57.m128i_i64[0];
        if ( !v46 )
          goto LABEL_3;
      }
      v32 = a1[1];
      v33 = *(_QWORD *)(v32 + 864);
      v34 = *(_DWORD *)(v32 + 848);
      v35 = *(_QWORD *)v32;
      v59 = 0;
      LODWORD(v60) = v34;
      if ( v35 )
      {
        if ( &v59 != (__int64 *)(v35 + 48) )
        {
          v36 = *(_QWORD *)(v35 + 48);
          v59 = v36;
          if ( v36 )
            sub_B96E90((__int64)&v59, v36, 1);
        }
      }
      v37 = v58.m128i_i32[0];
      v38 = 216;
      v39 = v58.m128i_i32[2];
      goto LABEL_25;
    }
    if ( (_WORD)v16 )
      goto LABEL_3;
LABEL_28:
    if ( v17 == v58.m128i_i64[1] )
      goto LABEL_3;
    goto LABEL_29;
  }
LABEL_37:
  if ( v31 != v48 )
    goto LABEL_11;
  v42 = a1[1];
  v33 = *(_QWORD *)(v42 + 864);
  v43 = *(_DWORD *)(v42 + 848);
  v44 = *(_QWORD *)v42;
  v59 = 0;
  LODWORD(v60) = v43;
  if ( v44 )
  {
    if ( &v59 != (__int64 *)(v44 + 48) )
    {
      v45 = *(_QWORD *)(v44 + 48);
      v59 = v45;
      if ( v45 )
        sub_B96E90((__int64)&v59, v45, 1);
    }
  }
  v37 = v58.m128i_i32[0];
  v38 = 234;
  v39 = v58.m128i_i32[2];
LABEL_25:
  v3 = sub_33FAF80(v33, v38, (unsigned int)&v59, v37, v39, v11);
  v12 = v40;
  if ( v59 )
  {
    v57.m128i_i64[0] = v40;
    sub_B91220((__int64)&v59, v59);
    v12 = v57.m128i_i64[0];
  }
LABEL_3:
  v18 = a1[3];
  v19 = _mm_load_si128(&v58);
  v20 = *(unsigned int *)(v18 + 8);
  if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
  {
    v47 = a1[3];
    v57 = v19;
    v56 = v12;
    sub_C8D5F0(v47, (const void *)(v18 + 16), v20 + 1, 0x10u, v12, v11);
    v20 = *(unsigned int *)(v18 + 8);
    v19 = _mm_load_si128(&v57);
    v12 = v56;
  }
  *(__m128i *)(*(_QWORD *)v18 + 16 * v20) = v19;
  ++*(_DWORD *)(v18 + 8);
  v21 = a1[4];
  v22 = v12 | a3 & 0xFFFFFFFF00000000LL;
  v23 = *(unsigned int *)(v21 + 8);
  if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(v21 + 12) )
  {
    sub_C8D5F0(v21, (const void *)(v21 + 16), v23 + 1, 0x10u, v12, v11);
    v23 = *(unsigned int *)(v21 + 8);
  }
  result = (__int64 *)(*(_QWORD *)v21 + 16 * v23);
  *result = v3;
  result[1] = v22;
  ++*(_DWORD *)(v21 + 8);
  return result;
}
