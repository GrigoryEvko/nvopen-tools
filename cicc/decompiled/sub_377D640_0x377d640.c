// Function: sub_377D640
// Address: 0x377d640
//
void __fastcall sub_377D640(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rsi
  __int64 v9; // rsi
  __int16 *v10; // rax
  __int16 v11; // dx
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rdi
  __m128i v15; // xmm0
  unsigned int v16; // r13d
  int v17; // r9d
  int v18; // edx
  __int64 v19; // rsi
  unsigned __int16 *v20; // rax
  unsigned int v21; // ecx
  __int64 v22; // r13
  unsigned int v23; // r12d
  unsigned __int64 v24; // r8
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  unsigned __int8 *v28; // r12
  __int64 v29; // r13
  __int64 v30; // r10
  __int64 v31; // r9
  int v32; // eax
  unsigned int v33; // ecx
  unsigned int v34; // edx
  unsigned __int64 v35; // r13
  unsigned __int8 *v36; // r12
  unsigned int v37; // edx
  unsigned __int64 v38; // r13
  unsigned __int8 *v39; // rax
  __int64 v40; // rcx
  __int64 v41; // r8
  int v42; // edx
  __int64 v43; // r9
  unsigned __int8 *v44; // rax
  bool v45; // cc
  int v46; // edx
  __int64 v47; // rdx
  __int128 v48; // [rsp-10h] [rbp-150h]
  unsigned __int64 v49; // [rsp+8h] [rbp-138h]
  __int16 v50; // [rsp+12h] [rbp-12Eh]
  unsigned int v51; // [rsp+18h] [rbp-128h]
  __int64 v52; // [rsp+28h] [rbp-118h]
  __int64 v53; // [rsp+28h] [rbp-118h]
  __int64 v54; // [rsp+28h] [rbp-118h]
  __m128i v55; // [rsp+A0h] [rbp-A0h] BYREF
  __m128i v56; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v57; // [rsp+C0h] [rbp-80h] BYREF
  int v58; // [rsp+C8h] [rbp-78h]
  unsigned __int64 v59; // [rsp+D0h] [rbp-70h] BYREF
  unsigned int v60; // [rsp+D8h] [rbp-68h]
  unsigned __int64 v61; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v62; // [rsp+E8h] [rbp-58h]
  __m128i v63; // [rsp+F0h] [rbp-50h] BYREF
  __m128i v64[4]; // [rsp+100h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a2 + 80);
  v55.m128i_i16[0] = 0;
  v55.m128i_i64[1] = 0;
  v56.m128i_i16[0] = 0;
  v56.m128i_i64[1] = 0;
  v57 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v57, v8, 1);
  v9 = *(_QWORD *)(a1 + 8);
  v58 = *(_DWORD *)(a2 + 72);
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOWORD(v61) = v11;
  v62 = v12;
  sub_33D0340((__int64)&v63, v9, (__int64 *)&v61);
  v13 = *(__int64 **)(a2 + 40);
  v14 = *(_QWORD *)(a1 + 8);
  v15 = _mm_loadu_si128(&v63);
  v55 = v15;
  v56 = _mm_loadu_si128(v64);
  v16 = *((_DWORD *)v13 + 2);
  v52 = *v13;
  *(_QWORD *)a3 = sub_33FAF80(v14, 170, (__int64)&v57, v15.m128i_u32[0], v15.m128i_i64[1], v17, v15);
  *(_DWORD *)(a3 + 8) = v18;
  v19 = *(_QWORD *)(v52 + 96);
  v20 = (unsigned __int16 *)(*(_QWORD *)(v52 + 48) + 16LL * v16);
  v22 = *((_QWORD *)v20 + 1);
  v23 = *v20;
  v60 = *(_DWORD *)(v19 + 32);
  v21 = v60;
  if ( v60 > 0x40 )
  {
    sub_C43780((__int64)&v59, (const void **)(v19 + 24));
    v21 = v60;
  }
  else
  {
    v59 = *(_QWORD *)(v19 + 24);
  }
  v53 = *(_QWORD *)(a1 + 8);
  if ( !v55.m128i_i16[0] )
  {
    v51 = v21;
    v24 = (unsigned int)sub_3007240((__int64)&v55);
    LODWORD(v62) = v51;
    if ( v51 <= 0x40 )
      goto LABEL_7;
LABEL_23:
    v49 = v24;
    sub_C43780((__int64)&v61, (const void **)&v59);
    v24 = v49;
    goto LABEL_8;
  }
  LODWORD(v62) = v21;
  v24 = word_4456340[v55.m128i_u16[0] - 1];
  if ( v21 > 0x40 )
    goto LABEL_23;
LABEL_7:
  v61 = v59;
LABEL_8:
  sub_C47170((__int64)&v61, v24);
  v63.m128i_i32[2] = v62;
  LODWORD(v62) = 0;
  v63.m128i_i64[0] = v61;
  v28 = sub_3401900(v53, (__int64)&v57, v23, v22, (__int64)&v63, 1, v15);
  v29 = v25;
  if ( v63.m128i_i32[2] > 0x40u && v63.m128i_i64[0] )
    j_j___libc_free_0_0(v63.m128i_u64[0]);
  if ( (unsigned int)v62 > 0x40 && v61 )
    j_j___libc_free_0_0(v61);
  v30 = *(_QWORD *)(a1 + 8);
  if ( v56.m128i_i16[0] )
  {
    v31 = 0;
    LOWORD(v32) = word_4456580[v56.m128i_u16[0] - 1];
  }
  else
  {
    v54 = *(_QWORD *)(a1 + 8);
    v32 = sub_3009970((__int64)&v56, (__int64)&v57, v25, v26, v27);
    v30 = v54;
    v50 = HIWORD(v32);
    v31 = v47;
  }
  HIWORD(v33) = v50;
  LOWORD(v33) = v32;
  sub_33FB160(v30, (__int64)v28, v29, (__int64)&v57, v33, v31, v15);
  v35 = v34 | v29 & 0xFFFFFFFF00000000LL;
  v36 = sub_33FAF80(*(_QWORD *)(a1 + 8), 168, (__int64)&v57, v56.m128i_u32[0], v56.m128i_i64[1], 0, v15);
  v38 = v37 | v35 & 0xFFFFFFFF00000000LL;
  v39 = sub_33FAF80(*(_QWORD *)(a1 + 8), 170, (__int64)&v57, v56.m128i_u32[0], v56.m128i_i64[1], 0, v15);
  v40 = v56.m128i_u32[0];
  v41 = v56.m128i_i64[1];
  *(_QWORD *)a4 = v39;
  *((_QWORD *)&v48 + 1) = v38;
  *(_DWORD *)(a4 + 8) = v42;
  *(_QWORD *)&v48 = v36;
  v44 = sub_3406EB0(*(_QWORD **)(a1 + 8), 0x38u, (__int64)&v57, v40, v41, v43, *(_OWORD *)a4, v48);
  v45 = v60 <= 0x40;
  *(_QWORD *)a4 = v44;
  *(_DWORD *)(a4 + 8) = v46;
  if ( !v45 && v59 )
    j_j___libc_free_0_0(v59);
  if ( v57 )
    sub_B91220((__int64)&v57, v57);
}
