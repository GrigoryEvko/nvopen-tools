// Function: sub_36DDC60
// Address: 0x36ddc60
//
__int64 __fastcall sub_36DDC60(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int32 v7; // ecx
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // rax
  unsigned __int16 v11; // di
  int v12; // eax
  unsigned int v13; // r13d
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // r10
  unsigned __int16 v17; // r11
  int v18; // r13d
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // r10
  __int64 v22; // rax
  _QWORD *v23; // rsi
  unsigned __int8 *v24; // rax
  __int32 v25; // edx
  __int64 v26; // r8
  __int64 v27; // r9
  __m128i *v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // r9
  __int64 v31; // r13
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v36; // rax
  __int64 v37; // r10
  unsigned __int16 v38; // r11
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // r10
  __int16 v43; // r11
  _QWORD *v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // [rsp-10h] [rbp-110h]
  unsigned __int64 v48; // [rsp+0h] [rbp-100h]
  __int64 v49; // [rsp+8h] [rbp-F8h]
  __int64 v50; // [rsp+10h] [rbp-F0h]
  __int32 v51; // [rsp+18h] [rbp-E8h]
  __int32 v52; // [rsp+1Ch] [rbp-E4h]
  __int64 v53; // [rsp+30h] [rbp-D0h] BYREF
  int v54; // [rsp+38h] [rbp-C8h]
  __int64 v55; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v56; // [rsp+48h] [rbp-B8h]
  _BYTE v57[32]; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v58; // [rsp+70h] [rbp-90h] BYREF
  __m128i v59; // [rsp+80h] [rbp-80h] BYREF
  __m128i v60; // [rsp+90h] [rbp-70h] BYREF
  __int16 v61; // [rsp+A0h] [rbp-60h]
  __int64 v62; // [rsp+A8h] [rbp-58h]
  __int16 v63; // [rsp+B0h] [rbp-50h]
  __int64 v64; // [rsp+B8h] [rbp-48h]
  __int16 v65; // [rsp+C0h] [rbp-40h]
  __int64 v66; // [rsp+C8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = *(_DWORD *)(v5 + 8);
  v8 = *(_QWORD *)v5;
  v53 = v6;
  v9 = *(_QWORD *)(v5 + 80);
  v52 = v7;
  v50 = *(_QWORD *)(v5 + 120);
  v51 = *(_DWORD *)(v5 + 128);
  if ( v6 )
    sub_B96E90((__int64)&v53, v6, 1);
  v54 = *(_DWORD *)(a2 + 72);
  v10 = (unsigned int)(*(_DWORD *)(a2 + 24) - 568);
  if ( (unsigned int)v10 > 2 )
    goto LABEL_14;
  v11 = *(_WORD *)(a2 + 96);
  v12 = dword_4501008[v10];
  if ( v12 == 2 )
  {
    v58.m128i_i64[0] = 0x100000AABLL;
    v55 = 0x100000AAELL;
    v36 = sub_36D6650(v11, 2735, 2732, 2733, 0x100000AAELL, 2730, 0x100000AABLL);
    v18 = v36;
    if ( BYTE4(v36) )
    {
      v39 = sub_33E48F0(*(_QWORD **)(a1 + 64), v38, v37, v38, v37, v47, 1, 0, 262, 0);
      v49 = v40;
      v21 = v39;
      goto LABEL_9;
    }
LABEL_14:
    v13 = 0;
    goto LABEL_15;
  }
  if ( v12 == 4 )
  {
    v58.m128i_i8[4] = 0;
    BYTE4(v55) = 0;
    v41 = sub_36D6650(v11, 2739, 2737, 2738, v55, 2736, v58.m128i_i64[0]);
    v18 = v41;
    if ( BYTE4(v41) )
    {
      v44 = *(_QWORD **)(a1 + 64);
      v65 = 262;
      v58.m128i_i64[1] = v42;
      v59.m128i_i64[1] = v42;
      v60.m128i_i64[1] = v42;
      v62 = v42;
      v58.m128i_i16[0] = v43;
      v59.m128i_i16[0] = v43;
      v60.m128i_i16[0] = v43;
      v61 = v43;
      v63 = 1;
      v64 = 0;
      v66 = 0;
      v45 = sub_33E5830(v44, (unsigned __int16 *)&v58, 6);
      v49 = v46;
      v21 = v45;
      goto LABEL_9;
    }
    goto LABEL_14;
  }
  v13 = 0;
  if ( v12 != 1 )
    goto LABEL_15;
  v58.m128i_i64[0] = 0x100000AA5LL;
  v55 = 0x100000AA8LL;
  v14 = sub_36D6650(v11, 2729, 2726, 2727, 0x100000AA8LL, 2724, 0x100000AA5LL);
  v18 = v14;
  if ( !BYTE4(v14) )
    goto LABEL_14;
  v19 = sub_33E5B50(*(_QWORD **)(a1 + 64), v17, v16, 1, 0, v15, 262, 0);
  v49 = v20;
  v21 = v19;
LABEL_9:
  v22 = *(_QWORD *)(v9 + 96);
  v23 = *(_QWORD **)(v22 + 24);
  if ( *(_DWORD *)(v22 + 32) > 0x40u )
    v23 = (_QWORD *)*v23;
  v48 = v21;
  v24 = sub_3400BD0(*(_QWORD *)(a1 + 64), (unsigned int)v23, (__int64)&v53, 7, 0, 1u, a3, 0);
  v59.m128i_i64[0] = v8;
  v58.m128i_i32[2] = v25;
  v58.m128i_i64[0] = (__int64)v24;
  v55 = (__int64)v57;
  v59.m128i_i32[2] = v52;
  v60.m128i_i64[0] = v50;
  v60.m128i_i32[2] = v51;
  v56 = 0x200000000LL;
  sub_C8D5F0((__int64)&v55, v57, 3u, 0x10u, v26, v27);
  v28 = (__m128i *)(v55 + 16LL * (unsigned int)v56);
  *v28 = _mm_loadu_si128(&v58);
  v28[1] = _mm_loadu_si128(&v59);
  v28[2] = _mm_loadu_si128(&v60);
  v29 = *(_QWORD **)(a1 + 64);
  LODWORD(v56) = v56 + 3;
  v31 = sub_33E66D0(v29, v18, (__int64)&v53, v48, v49, v30, (unsigned __int64 *)v55, (unsigned int)v56);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v31, v32, v33, v34);
  sub_3421DB0(v31);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( (_BYTE *)v55 != v57 )
    _libc_free(v55);
  v13 = 1;
LABEL_15:
  if ( v53 )
    sub_B91220((__int64)&v53, v53);
  return v13;
}
