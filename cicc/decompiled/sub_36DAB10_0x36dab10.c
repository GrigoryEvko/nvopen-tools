// Function: sub_36DAB10
// Address: 0x36dab10
//
void __fastcall sub_36DAB10(__int64 a1, __int64 a2, __int32 a3, __m128i a4)
{
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // rsi
  int v12; // eax
  __int64 v13; // r8
  __m128i v14; // xmm1
  int v15; // r13d
  __int64 v16; // rax
  _QWORD *v17; // rsi
  unsigned __int8 *v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r9
  unsigned __int64 v22; // rdx
  unsigned __int64 *v23; // rax
  const __m128i *v24; // rdx
  __m128i v25; // xmm0
  unsigned __int64 *v26; // rdx
  __int64 v27; // rax
  _QWORD *v28; // r9
  unsigned __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r13
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __m128i v35; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v36; // [rsp+10h] [rbp-90h] BYREF
  int v37; // [rsp+18h] [rbp-88h]
  unsigned __int64 *v38; // [rsp+20h] [rbp-80h] BYREF
  __int64 v39; // [rsp+28h] [rbp-78h]
  _OWORD v40[7]; // [rsp+30h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a2 + 80);
  v35.m128i_i32[0] = a3;
  v36 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v36, v6, 1);
  v7 = *(_QWORD *)(a2 + 40);
  v37 = *(_DWORD *)(a2 + 72);
  v8 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 96LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v10 = *(_QWORD *)(*(_QWORD *)(v7 + 80) + 96LL);
  v11 = *(_QWORD **)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = (_QWORD *)*v11;
  v12 = sub_36D6A40((int)v9, (unsigned __int8)v11 & 1, 1);
  v14 = _mm_loadu_si128((const __m128i *)(v13 + 120));
  v38 = (unsigned __int64 *)v40;
  v15 = v12;
  v39 = 0x400000001LL;
  v40[0] = v14;
  if ( v35.m128i_i8[0] )
  {
    v16 = *(_QWORD *)(*(_QWORD *)(v13 + 160) + 96LL);
    v17 = *(_QWORD **)(v16 + 24);
    if ( *(_DWORD *)(v16 + 32) > 0x40u )
      v17 = (_QWORD *)*v17;
    v18 = sub_3400BD0(*(_QWORD *)(a1 + 64), (__int64)v17, (__int64)&v36, 7, 0, 1u, a4, 0);
    v19 = (unsigned int)v39;
    v21 = v20;
    v22 = (unsigned int)v39 + 1LL;
    if ( v22 > HIDWORD(v39) )
    {
      v35.m128i_i64[0] = (__int64)v18;
      v35.m128i_i64[1] = v21;
      sub_C8D5F0((__int64)&v38, v40, v22, 0x10u, (__int64)v18, v21);
      v19 = (unsigned int)v39;
      v21 = v35.m128i_i64[1];
      v18 = (unsigned __int8 *)v35.m128i_i64[0];
    }
    v23 = &v38[2 * v19];
    *v23 = (unsigned __int64)v18;
    v23[1] = v21;
    v24 = *(const __m128i **)(a2 + 40);
    LODWORD(v39) = v39 + 1;
    v25 = _mm_loadu_si128(v24);
    if ( (unsigned __int64)(unsigned int)v39 + 1 > HIDWORD(v39) )
    {
      v35 = v25;
      sub_C8D5F0((__int64)&v38, v40, (unsigned int)v39 + 1LL, 0x10u, (__int64)v18, v21);
      v26 = v38;
      v25 = _mm_load_si128(&v35);
    }
    else
    {
      v26 = v38;
    }
    v27 = 2LL * (unsigned int)v39;
  }
  else
  {
    v25 = _mm_loadu_si128((const __m128i *)v13);
    v27 = 2;
    v26 = (unsigned __int64 *)v40;
  }
  *(__m128i *)&v26[v27] = v25;
  v28 = *(_QWORD **)(a1 + 64);
  v29 = *(_QWORD *)(a2 + 48);
  v30 = *(unsigned int *)(a2 + 68);
  LODWORD(v39) = v39 + 1;
  v31 = sub_33E66D0(v28, v15, (__int64)&v36, v29, v30, (__int64)v28, v38, (unsigned int)v39);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v31, v32, v33, v34);
  sub_3421DB0(v31);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v38 != (unsigned __int64 *)v40 )
    _libc_free((unsigned __int64)v38);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
}
