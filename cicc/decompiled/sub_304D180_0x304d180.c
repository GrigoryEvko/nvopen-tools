// Function: sub_304D180
// Address: 0x304d180
//
__int64 __fastcall sub_304D180(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // r14d
  __int64 v6; // rsi
  __m128i v7; // xmm0
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // rax
  __m128i v11; // xmm3
  __int64 v12; // rax
  __int32 v13; // edx
  __m128i v14; // xmm2
  __int64 v15; // r8
  __int64 v16; // r9
  __m128i *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r9
  __int64 v22; // r10
  __int64 v23; // r11
  int v24; // r8d
  __int64 v25; // rcx
  __int64 result; // rax
  __int64 v27; // [rsp+8h] [rbp-138h]
  __int64 v28; // [rsp+10h] [rbp-130h]
  __int64 v29; // [rsp+18h] [rbp-128h]
  int v30; // [rsp+20h] [rbp-120h]
  int v31; // [rsp+28h] [rbp-118h]
  __int64 v32; // [rsp+30h] [rbp-110h]
  __int64 v33; // [rsp+30h] [rbp-110h]
  __int64 v34; // [rsp+30h] [rbp-110h]
  __int64 v35; // [rsp+38h] [rbp-108h]
  __int64 v36; // [rsp+40h] [rbp-100h] BYREF
  int v37; // [rsp+48h] [rbp-F8h]
  __m128i v38; // [rsp+50h] [rbp-F0h] BYREF
  __m128i v39; // [rsp+60h] [rbp-E0h] BYREF
  _BYTE *v40; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v41; // [rsp+78h] [rbp-C8h]
  _BYTE v42[64]; // [rsp+80h] [rbp-C0h] BYREF
  __m128i v43; // [rsp+C0h] [rbp-80h] BYREF
  __m128i v44; // [rsp+D0h] [rbp-70h] BYREF
  __m128i v45; // [rsp+E0h] [rbp-60h] BYREF
  __m128i v46; // [rsp+F0h] [rbp-50h] BYREF
  __m128i v47[4]; // [rsp+100h] [rbp-40h] BYREF

  v5 = sub_3032270(a2, a2);
  sub_3030880((__int64)&v38, a2, a3);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v36 = v6;
  v44 = _mm_loadu_si128(&v38);
  v43 = v7;
  if ( v6 )
    sub_B96E90((__int64)&v36, v6, 1);
  v8 = *(_QWORD *)(a2 + 112);
  v37 = *(_DWORD *)(a2 + 72);
  v9 = sub_2EAC1E0(v8);
  v10 = sub_3400BD0(a3, v9, (unsigned int)&v36, 5, 0, 1, 0);
  v11 = _mm_loadu_si128(&v39);
  v45.m128i_i64[0] = v10;
  v12 = *(_QWORD *)(a2 + 40);
  v45.m128i_i32[2] = v13;
  v14 = _mm_loadu_si128((const __m128i *)(v12 + 120));
  v40 = v42;
  v41 = 0x400000000LL;
  v46 = v14;
  v47[0] = v11;
  sub_C8D5F0((__int64)&v40, v42, 5u, 0x10u, v15, v16);
  v17 = (__m128i *)&v40[16 * (unsigned int)v41];
  *v17 = _mm_loadu_si128(&v43);
  v18 = v36;
  v17[1] = _mm_loadu_si128(&v44);
  v17[2] = _mm_loadu_si128(&v45);
  v17[3] = _mm_loadu_si128(&v46);
  v17[4] = _mm_loadu_si128(v47);
  v19 = (unsigned int)(v41 + 5);
  LODWORD(v41) = v41 + 5;
  if ( v18 )
  {
    sub_B91220((__int64)&v36, v18);
    v19 = (unsigned int)v41;
  }
  v20 = *(_QWORD *)(a2 + 80);
  v21 = *(_QWORD *)(a2 + 112);
  v22 = *(unsigned __int16 *)(a2 + 96);
  v23 = *(_QWORD *)(a2 + 104);
  v35 = v19;
  v24 = *(_DWORD *)(a2 + 68);
  v43.m128i_i64[0] = v20;
  v32 = (__int64)v40;
  v25 = *(_QWORD *)(a2 + 48);
  if ( v20 )
  {
    v27 = *(_QWORD *)(a2 + 48);
    v30 = v24;
    v28 = v22;
    v29 = v23;
    v31 = v21;
    sub_B96E90((__int64)&v43, v20, 1);
    LODWORD(v25) = v27;
    v24 = v30;
    v22 = v28;
    v23 = v29;
    LODWORD(v21) = v31;
  }
  v43.m128i_i32[2] = *(_DWORD *)(a2 + 72);
  result = sub_33EA9D0(a3, v5, (unsigned int)&v43, v25, v24, v21, v32, v35, v22, v23);
  if ( v43.m128i_i64[0] )
  {
    v33 = result;
    sub_B91220((__int64)&v43, v43.m128i_i64[0]);
    result = v33;
  }
  if ( v40 != v42 )
  {
    v34 = result;
    _libc_free((unsigned __int64)v40);
    return v34;
  }
  return result;
}
