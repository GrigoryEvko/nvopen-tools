// Function: sub_36DA6A0
// Address: 0x36da6a0
//
void __fastcall sub_36DA6A0(__int64 a1, __int64 a2, int a3, __m128i a4)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  int v9; // r12d
  int v10; // r13d
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int8 *v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r9
  unsigned __int64 v18; // rdx
  unsigned __int64 *v19; // rax
  __int32 v20; // r11d
  unsigned __int64 **v21; // rdi
  __int64 v22; // r11
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // r8
  __m128i v26; // xmm0
  __m128i v27; // xmm0
  _QWORD *v28; // r9
  unsigned __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r12
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __m128i v35; // [rsp+0h] [rbp-480h] BYREF
  __m128i v36; // [rsp+10h] [rbp-470h] BYREF
  __int64 v37; // [rsp+20h] [rbp-460h]
  unsigned __int64 *v38; // [rsp+28h] [rbp-458h]
  __int64 v39; // [rsp+30h] [rbp-450h] BYREF
  int v40; // [rsp+38h] [rbp-448h]
  unsigned __int64 *v41; // [rsp+40h] [rbp-440h] BYREF
  __int64 v42; // [rsp+48h] [rbp-438h]
  _BYTE v43[1072]; // [rsp+50h] [rbp-430h] BYREF

  v7 = *(_QWORD *)(a2 + 80);
  v39 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v39, v7, 1);
  v40 = *(_DWORD *)(a2 + 72);
  v38 = (unsigned __int64 *)v43;
  v41 = (unsigned __int64 *)v43;
  v42 = 0x4000000000LL;
  v8 = (unsigned int)(a3 - 10654);
  if ( (unsigned int)v8 > 0x7F )
    BUG();
  v9 = a3 - 5044;
  v10 = byte_45010A0[v8];
  v36.m128i_i32[0] = byte_4501020[v8];
  v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL) + 96LL);
  v12 = *(_QWORD *)(v11 + 24);
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    v12 = *(_QWORD *)v12;
  v13 = (v12 >> 3) & 1;
  if ( v10 != 1 )
    LOBYTE(v13) = 0;
  v14 = sub_3400BD0(*(_QWORD *)(a1 + 64), v12 & 0xFFFFFFFFFFFFFFF7LL | (8 * (v13 & 1)), (__int64)&v39, 8, 0, 1u, a4, 0);
  v15 = (unsigned int)v42;
  v17 = v16;
  v18 = (unsigned int)v42 + 1LL;
  if ( v18 > HIDWORD(v42) )
  {
    v35.m128i_i64[0] = (__int64)v14;
    v35.m128i_i64[1] = v17;
    sub_C8D5F0((__int64)&v41, v38, v18, 0x10u, (__int64)v14, v17);
    v15 = (unsigned int)v42;
    v17 = v35.m128i_i64[1];
    v14 = (unsigned __int8 *)v35.m128i_i64[0];
  }
  v19 = &v41[2 * v15];
  v20 = v36.m128i_i32[0];
  v21 = &v41;
  *v19 = (unsigned __int64)v14;
  v19[1] = v17;
  v22 = (unsigned int)(v10 + v20);
  v23 = 120;
  v24 = (unsigned int)(v42 + 1);
  v25 = 40 * v22 + 160;
  LODWORD(v42) = v42 + 1;
  do
  {
    v26 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v23));
    if ( v24 + 1 > (unsigned __int64)HIDWORD(v42) )
    {
      v37 = v25;
      v36.m128i_i64[0] = (__int64)v21;
      v35 = v26;
      sub_C8D5F0((__int64)v21, v38, v24 + 1, 0x10u, v25, v17);
      v24 = (unsigned int)v42;
      v25 = v37;
      v26 = _mm_load_si128(&v35);
      v21 = (unsigned __int64 **)v36.m128i_i64[0];
    }
    v23 += 40;
    *(__m128i *)&v41[2 * v24] = v26;
    v24 = (unsigned int)(v42 + 1);
    LODWORD(v42) = v42 + 1;
  }
  while ( v23 != v25 );
  v27 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  if ( v24 + 1 > (unsigned __int64)HIDWORD(v42) )
  {
    v36 = v27;
    sub_C8D5F0((__int64)&v41, v38, v24 + 1, 0x10u, v25, v17);
    v24 = (unsigned int)v42;
    v27 = _mm_load_si128(&v36);
  }
  *(__m128i *)&v41[2 * v24] = v27;
  v28 = *(_QWORD **)(a1 + 64);
  v29 = *(_QWORD *)(a2 + 48);
  v30 = *(unsigned int *)(a2 + 68);
  LODWORD(v42) = v42 + 1;
  v31 = sub_33E66D0(v28, v9, (__int64)&v39, v29, v30, (__int64)v28, v41, (unsigned int)v42);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v31, v32, v33, v34);
  sub_3421DB0(v31);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v41 != v38 )
    _libc_free((unsigned __int64)v41);
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
}
