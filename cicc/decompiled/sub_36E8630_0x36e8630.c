// Function: sub_36E8630
// Address: 0x36e8630
//
void __fastcall sub_36E8630(__int64 a1, unsigned int a2, int a3, __int64 a4, __m128i a5)
{
  unsigned int v5; // eax
  __int64 v10; // rsi
  __int64 v11; // rdx
  int v12; // edi
  __int64 v13; // rcx
  int v14; // eax
  __int64 v15; // rax
  _QWORD *v16; // rcx
  __int64 v17; // rdx
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned __int8 *v21; // r8
  __int64 v22; // rax
  unsigned int v23; // edx
  __int64 v24; // r9
  unsigned __int64 v25; // rdx
  unsigned __int64 *v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rdi
  unsigned __int8 *v29; // r8
  __int64 v30; // rax
  unsigned int v31; // edx
  __int64 v32; // r9
  unsigned __int64 v33; // rdx
  unsigned __int64 *v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rdi
  __int64 v37; // r9
  unsigned __int8 *v38; // r12
  __int64 v39; // rax
  unsigned int v40; // edx
  __int64 v41; // r8
  unsigned __int64 v42; // rdx
  unsigned __int64 *v43; // rax
  __int64 v44; // rax
  int v45; // edx
  __int64 v46; // r12
  unsigned __int64 **v47; // rdi
  __int64 v48; // r14
  __m128i v49; // xmm0
  __m128i v50; // xmm0
  _QWORD *v51; // r9
  unsigned __int64 v52; // rcx
  __int64 v53; // r12
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __m128i v57; // [rsp+0h] [rbp-290h] BYREF
  __m128i v58; // [rsp+10h] [rbp-280h] BYREF
  unsigned __int8 *v59; // [rsp+20h] [rbp-270h]
  unsigned __int64 *v60; // [rsp+28h] [rbp-268h]
  __int64 v61; // [rsp+30h] [rbp-260h] BYREF
  int v62; // [rsp+38h] [rbp-258h]
  __int64 v63; // [rsp+40h] [rbp-250h] BYREF
  int v64; // [rsp+48h] [rbp-248h]
  unsigned __int64 *v65; // [rsp+50h] [rbp-240h] BYREF
  __int64 v66; // [rsp+58h] [rbp-238h]
  _BYTE v67[560]; // [rsp+60h] [rbp-230h] BYREF

  v5 = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL);
  if ( v5 <= 0x47 || a2 > 1 && v5 == 72 )
    sub_C64ED0("immamma is not supported on this architecture", 1u);
  v10 = *(_QWORD *)(a4 + 80);
  v61 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v61, v10, 1);
  v11 = *(_QWORD *)(a4 + 40);
  v12 = *(_DWORD *)(a4 + 72);
  v13 = *(_QWORD *)(v11 + 80);
  v62 = v12;
  v14 = *(_DWORD *)(v13 + 24);
  if ( v14 != 11 && v14 != 35 )
    sub_C64ED0("rowcol not constant", 1u);
  v15 = *(_QWORD *)(v13 + 96);
  v16 = *(_QWORD **)(v15 + 24);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  v17 = *(_QWORD *)(v11 + 120);
  v18 = *(_DWORD *)(v17 + 24);
  if ( v18 != 35 && v18 != 11 )
    sub_C64ED0("satf not constant", 1u);
  v19 = *(_QWORD *)(v17 + 96);
  if ( *(_DWORD *)(v19 + 32) <= 0x40u )
    v58.m128i_i64[0] = *(_QWORD *)(v19 + 24);
  else
    v58.m128i_i64[0] = **(_QWORD **)(v19 + 24);
  v20 = *(_QWORD *)(a4 + 80);
  v60 = (unsigned __int64 *)v67;
  v65 = (unsigned __int64 *)v67;
  v66 = 0x2000000000LL;
  v63 = v20;
  if ( v20 )
  {
    v57.m128i_i64[0] = (__int64)v16;
    sub_B96E90((__int64)&v63, v20, 1);
    v12 = *(_DWORD *)(a4 + 72);
    LODWORD(v16) = v57.m128i_i32[0];
  }
  v64 = v12;
  v21 = sub_3400BD0(*(_QWORD *)(a1 + 64), (unsigned int)v16, (__int64)&v63, 7, 0, 1u, a5, 0);
  v22 = (unsigned int)v66;
  v24 = v23;
  v25 = (unsigned int)v66 + 1LL;
  if ( v25 > HIDWORD(v66) )
  {
    v59 = v21;
    v57.m128i_i64[0] = v24;
    sub_C8D5F0((__int64)&v65, v60, v25, 0x10u, (__int64)v21, v24);
    v22 = (unsigned int)v66;
    v21 = v59;
    v24 = v57.m128i_i64[0];
  }
  v26 = &v65[2 * v22];
  *v26 = (unsigned __int64)v21;
  v26[1] = v24;
  LODWORD(v66) = v66 + 1;
  if ( v63 )
    sub_B91220((__int64)&v63, v63);
  v27 = *(_QWORD *)(a4 + 80);
  v63 = v27;
  if ( v27 )
    sub_B96E90((__int64)&v63, v27, 1);
  v28 = *(_QWORD *)(a1 + 64);
  v64 = *(_DWORD *)(a4 + 72);
  v29 = sub_3400BD0(v28, v58.m128i_u32[0], (__int64)&v63, 7, 0, 1u, a5, 0);
  v30 = (unsigned int)v66;
  v32 = v31;
  v33 = (unsigned int)v66 + 1LL;
  if ( v33 > HIDWORD(v66) )
  {
    v57.m128i_i64[0] = (__int64)v29;
    v58.m128i_i64[0] = v32;
    sub_C8D5F0((__int64)&v65, v60, v33, 0x10u, (__int64)v29, v32);
    v30 = (unsigned int)v66;
    v29 = (unsigned __int8 *)v57.m128i_i64[0];
    v32 = v58.m128i_i64[0];
  }
  v34 = &v65[2 * v30];
  *v34 = (unsigned __int64)v29;
  v34[1] = v32;
  LODWORD(v66) = v66 + 1;
  if ( v63 )
    sub_B91220((__int64)&v63, v63);
  v35 = *(_QWORD *)(a4 + 80);
  v63 = v35;
  if ( v35 )
    sub_B96E90((__int64)&v63, v35, 1);
  v36 = *(_QWORD *)(a1 + 64);
  v64 = *(_DWORD *)(a4 + 72);
  v38 = sub_3400BD0(v36, a2, (__int64)&v63, 7, 0, 1u, a5, 0);
  v39 = (unsigned int)v66;
  v41 = v40;
  v42 = (unsigned int)v66 + 1LL;
  if ( v42 > HIDWORD(v66) )
  {
    v58.m128i_i64[0] = v41;
    sub_C8D5F0((__int64)&v65, v60, v42, 0x10u, v41, v37);
    v39 = (unsigned int)v66;
    v41 = v58.m128i_i64[0];
  }
  v43 = &v65[2 * v39];
  *v43 = (unsigned __int64)v38;
  v43[1] = v41;
  v44 = (unsigned int)(v66 + 1);
  LODWORD(v66) = v66 + 1;
  if ( v63 )
  {
    sub_B91220((__int64)&v63, v63);
    v44 = (unsigned int)v66;
  }
  v45 = 12;
  if ( a3 != 1570 )
    v45 = 9 * (a3 != 1595) + 4;
  v46 = 160;
  v47 = &v65;
  v48 = 40LL * (unsigned int)(v45 - 1) + 200;
  do
  {
    v49 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 40) + v46));
    if ( v44 + 1 > (unsigned __int64)HIDWORD(v66) )
    {
      v58.m128i_i64[0] = (__int64)v47;
      v57 = v49;
      sub_C8D5F0((__int64)v47, v60, v44 + 1, 0x10u, v41, v37);
      v44 = (unsigned int)v66;
      v49 = _mm_load_si128(&v57);
      v47 = (unsigned __int64 **)v58.m128i_i64[0];
    }
    v46 += 40;
    *(__m128i *)&v65[2 * v44] = v49;
    v44 = (unsigned int)(v66 + 1);
    LODWORD(v66) = v66 + 1;
  }
  while ( v48 != v46 );
  v50 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a4 + 40));
  if ( v44 + 1 > (unsigned __int64)HIDWORD(v66) )
  {
    v58 = v50;
    sub_C8D5F0((__int64)&v65, v60, v44 + 1, 0x10u, v41, v37);
    v44 = (unsigned int)v66;
    v50 = _mm_load_si128(&v58);
  }
  *(__m128i *)&v65[2 * v44] = v50;
  v51 = *(_QWORD **)(a1 + 64);
  v52 = *(_QWORD *)(a4 + 48);
  LODWORD(v66) = v66 + 1;
  v53 = sub_33E66D0(v51, a3, (__int64)&v61, v52, *(unsigned int *)(a4 + 68), (__int64)v51, v65, (unsigned int)v66);
  sub_34158F0(*(_QWORD *)(a1 + 64), a4, v53, v54, v55, v56);
  sub_3421DB0(v53);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a4);
  if ( v65 != v60 )
    _libc_free((unsigned __int64)v65);
  if ( v61 )
    sub_B91220((__int64)&v61, v61);
}
