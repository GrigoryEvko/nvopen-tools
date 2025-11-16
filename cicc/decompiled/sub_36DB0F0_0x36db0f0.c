// Function: sub_36DB0F0
// Address: 0x36db0f0
//
void __fastcall sub_36DB0F0(__int64 a1, __int64 a2, __int32 a3, __m128i a4)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // r12
  __int64 v10; // rax
  int v11; // r12d
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned __int8 *v15; // r8
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r9
  unsigned __int64 v19; // rdx
  unsigned __int64 *v20; // rax
  __int64 v21; // rax
  __m128i v22; // xmm0
  __int64 v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int64 v28; // rdx
  unsigned __int64 *v29; // rax
  __m128i v30; // xmm0
  _QWORD *v31; // r9
  unsigned __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r12
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __m128i v38; // [rsp+0h] [rbp-B0h] BYREF
  __m128i v39; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+20h] [rbp-90h] BYREF
  int v41; // [rsp+28h] [rbp-88h]
  unsigned __int64 *v42; // [rsp+30h] [rbp-80h] BYREF
  __int64 v43; // [rsp+38h] [rbp-78h]
  _BYTE v44[112]; // [rsp+40h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a2 + 80);
  v39.m128i_i32[0] = a3;
  v40 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v40, v6, 1);
  v41 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 96LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  if ( (unsigned int)((_DWORD)v9 - 10257) > 0x29 )
    BUG();
  v10 = *(_QWORD *)(v7 + 80);
  v11 = (_DWORD)v9 - 7037;
  v42 = (unsigned __int64 *)v44;
  v12 = *(_QWORD *)(a1 + 64);
  v13 = *(_QWORD *)(v10 + 96);
  v43 = 0x400000000LL;
  if ( *(_DWORD *)(v13 + 32) <= 0x40u )
    v14 = *(_QWORD *)(v13 + 24);
  else
    v14 = **(_QWORD **)(v13 + 24);
  v15 = sub_3400BD0(v12, v14, (__int64)&v40, 7, 0, 1u, a4, 0);
  v16 = (unsigned int)v43;
  v18 = v17;
  v19 = (unsigned int)v43 + 1LL;
  if ( v19 > HIDWORD(v43) )
  {
    v38.m128i_i64[0] = (__int64)v15;
    v38.m128i_i64[1] = v18;
    sub_C8D5F0((__int64)&v42, v44, v19, 0x10u, (__int64)v15, v18);
    v16 = (unsigned int)v43;
    v18 = v38.m128i_i64[1];
    v15 = (unsigned __int8 *)v38.m128i_i64[0];
  }
  v20 = &v42[2 * v16];
  *v20 = (unsigned __int64)v15;
  v20[1] = v18;
  v22 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 120LL));
  LODWORD(v43) = v43 + 1;
  v21 = (unsigned int)v43;
  if ( (unsigned __int64)(unsigned int)v43 + 1 > HIDWORD(v43) )
  {
    v38 = v22;
    sub_C8D5F0((__int64)&v42, v44, (unsigned int)v43 + 1LL, 0x10u, (__int64)v15, v18);
    v21 = (unsigned int)v43;
    v22 = _mm_load_si128(&v38);
  }
  *(__m128i *)&v42[2 * v21] = v22;
  v23 = (unsigned int)(v43 + 1);
  LODWORD(v43) = v43 + 1;
  if ( v39.m128i_i8[0] )
  {
    v24 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 160LL) + 96LL);
    v25 = *(_QWORD **)(v24 + 24);
    if ( *(_DWORD *)(v24 + 32) > 0x40u )
      v25 = (_QWORD *)*v25;
    v15 = sub_3400BD0(*(_QWORD *)(a1 + 64), (__int64)v25, (__int64)&v40, 7, 0, 1u, v22, 0);
    v26 = (unsigned int)v43;
    v18 = v27;
    v28 = (unsigned int)v43 + 1LL;
    if ( v28 > HIDWORD(v43) )
    {
      v39.m128i_i64[0] = (__int64)v15;
      v39.m128i_i64[1] = v18;
      sub_C8D5F0((__int64)&v42, v44, v28, 0x10u, (__int64)v15, v18);
      v26 = (unsigned int)v43;
      v18 = v39.m128i_i64[1];
      v15 = (unsigned __int8 *)v39.m128i_i64[0];
    }
    v29 = &v42[2 * v26];
    *v29 = (unsigned __int64)v15;
    v29[1] = v18;
    v23 = (unsigned int)(v43 + 1);
    LODWORD(v43) = v43 + 1;
  }
  v30 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  if ( v23 + 1 > (unsigned __int64)HIDWORD(v43) )
  {
    v39 = v30;
    sub_C8D5F0((__int64)&v42, v44, v23 + 1, 0x10u, (__int64)v15, v18);
    v23 = (unsigned int)v43;
    v30 = _mm_load_si128(&v39);
  }
  *(__m128i *)&v42[2 * v23] = v30;
  v31 = *(_QWORD **)(a1 + 64);
  v32 = *(_QWORD *)(a2 + 48);
  v33 = *(unsigned int *)(a2 + 68);
  LODWORD(v43) = v43 + 1;
  v34 = sub_33E66D0(v31, v11, (__int64)&v40, v32, v33, (__int64)v31, v42, (unsigned int)v43);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v34, v35, v36, v37);
  sub_3421DB0(v34);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v42 != (unsigned __int64 *)v44 )
    _libc_free((unsigned __int64)v42);
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
}
