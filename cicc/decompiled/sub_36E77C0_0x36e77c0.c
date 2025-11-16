// Function: sub_36E77C0
// Address: 0x36e77c0
//
void __fastcall sub_36E77C0(__int64 a1, __int32 a2, int a3, __int64 a4, __m128i a5)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  int v10; // eax
  __int64 v11; // rcx
  int v12; // edx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rdi
  unsigned __int8 *v17; // r8
  __int64 v18; // rax
  unsigned int v19; // edx
  __int64 v20; // r9
  unsigned __int64 v21; // rdx
  unsigned __int64 *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // r14
  unsigned __int64 **v26; // rdi
  __int64 v27; // r8
  __m128i v28; // xmm0
  __m128i v29; // xmm0
  _QWORD *v30; // r9
  unsigned __int64 v31; // rcx
  __int64 v32; // r13
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // [rsp+8h] [rbp-288h]
  unsigned __int8 *v37; // [rsp+8h] [rbp-288h]
  __m128i v38; // [rsp+10h] [rbp-280h] BYREF
  __m128i v39; // [rsp+20h] [rbp-270h] BYREF
  __int64 v40; // [rsp+30h] [rbp-260h] BYREF
  int v41; // [rsp+38h] [rbp-258h]
  __int64 v42; // [rsp+40h] [rbp-250h] BYREF
  int v43; // [rsp+48h] [rbp-248h]
  unsigned __int64 *v44; // [rsp+50h] [rbp-240h] BYREF
  __int64 v45; // [rsp+58h] [rbp-238h]
  _BYTE v46[560]; // [rsp+60h] [rbp-230h] BYREF

  v5 = *(_QWORD *)(a1 + 1136);
  v39.m128i_i32[0] = a2;
  if ( *(_DWORD *)(v5 + 344) <= 0x45u )
    sub_C64ED0("hmmamma is not supported on this architecture", 1u);
  v6 = *(_QWORD *)(a4 + 80);
  v40 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v40, v6, 1);
  v10 = *(_DWORD *)(a4 + 72);
  v11 = *(_QWORD *)(*(_QWORD *)(a4 + 40) + 80LL);
  v41 = v10;
  v12 = *(_DWORD *)(v11 + 24);
  if ( v12 != 11 && v12 != 35 )
    sub_C64ED0("rowcol not constant", 1u);
  v13 = *(_QWORD *)(v11 + 96);
  v14 = *(_QWORD *)(v13 + 24);
  if ( *(_DWORD *)(v13 + 32) > 0x40u )
    v14 = **(_QWORD **)(v13 + 24);
  v15 = *(_QWORD *)(a4 + 80);
  v44 = (unsigned __int64 *)v46;
  v45 = 0x2000000000LL;
  v42 = v15;
  if ( v15 )
  {
    v38.m128i_i64[0] = v14;
    sub_B96E90((__int64)&v42, v15, 1);
    v10 = *(_DWORD *)(a4 + 72);
    LODWORD(v14) = v38.m128i_i32[0];
  }
  v16 = *(_QWORD *)(a1 + 64);
  v43 = v10;
  v17 = sub_3400BD0(v16, (unsigned int)v14, (__int64)&v42, 7, 0, 1u, a5, 0);
  v18 = (unsigned int)v45;
  v20 = v19;
  v21 = (unsigned int)v45 + 1LL;
  if ( v21 > HIDWORD(v45) )
  {
    v37 = v17;
    v38.m128i_i64[0] = v20;
    sub_C8D5F0((__int64)&v44, v46, v21, 0x10u, (__int64)v17, v20);
    v18 = (unsigned int)v45;
    v17 = v37;
    v20 = v38.m128i_i64[0];
  }
  v22 = &v44[2 * v18];
  *v22 = (unsigned __int64)v17;
  v23 = v42;
  v22[1] = v20;
  v24 = (unsigned int)(v45 + 1);
  LODWORD(v45) = v45 + 1;
  if ( v23 )
  {
    sub_B91220((__int64)&v42, v23);
    v24 = (unsigned int)v45;
  }
  v25 = 160;
  v26 = &v44;
  v27 = 40LL * (v39.m128i_i8[0] == 0 ? 19 : 23) + 200;
  do
  {
    v28 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 40) + v25));
    if ( v24 + 1 > (unsigned __int64)HIDWORD(v45) )
    {
      v36 = v27;
      v39.m128i_i64[0] = (__int64)v26;
      v38 = v28;
      sub_C8D5F0((__int64)v26, v46, v24 + 1, 0x10u, v27, v20);
      v24 = (unsigned int)v45;
      v27 = v36;
      v28 = _mm_load_si128(&v38);
      v26 = (unsigned __int64 **)v39.m128i_i64[0];
    }
    v25 += 40;
    *(__m128i *)&v44[2 * v24] = v28;
    v24 = (unsigned int)(v45 + 1);
    LODWORD(v45) = v45 + 1;
  }
  while ( v27 != v25 );
  v29 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a4 + 40));
  if ( v24 + 1 > (unsigned __int64)HIDWORD(v45) )
  {
    v39 = v29;
    sub_C8D5F0((__int64)&v44, v46, v24 + 1, 0x10u, v27, v20);
    v24 = (unsigned int)v45;
    v29 = _mm_load_si128(&v39);
  }
  *(__m128i *)&v44[2 * v24] = v29;
  v30 = *(_QWORD **)(a1 + 64);
  v31 = *(_QWORD *)(a4 + 48);
  LODWORD(v45) = v45 + 1;
  v32 = sub_33E66D0(v30, a3, (__int64)&v40, v31, *(unsigned int *)(a4 + 68), (__int64)v30, v44, (unsigned int)v45);
  sub_34158F0(*(_QWORD *)(a1 + 64), a4, v32, v33, v34, v35);
  sub_3421DB0(v32);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a4);
  if ( v44 != (unsigned __int64 *)v46 )
    _libc_free((unsigned __int64)v44);
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
}
