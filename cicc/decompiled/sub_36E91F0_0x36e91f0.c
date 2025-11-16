// Function: sub_36E91F0
// Address: 0x36e91f0
//
void __fastcall sub_36E91F0(__int64 a1, char a2, int a3, __int64 a4, __m128i a5)
{
  __int64 v6; // rsi
  __int64 v10; // rdi
  unsigned int v11; // eax
  const __m128i *v12; // rdx
  unsigned int v13; // r8d
  __int64 v14; // rcx
  int v15; // eax
  __int64 v16; // rax
  _QWORD *v17; // r15
  __int64 v18; // rsi
  __m128i v19; // xmm0
  __int64 v20; // rdi
  int v21; // edx
  __int64 v22; // rsi
  __m128i v23; // xmm1
  __int64 v24; // rdi
  __int64 v25; // r9
  int v26; // edx
  __int64 v27; // rax
  _QWORD *v28; // rdi
  __int64 v29; // rdx
  __m128i v30; // xmm3
  __int64 v31; // rdi
  int v32; // edx
  __int64 v33; // rsi
  __m128i v34; // xmm4
  __int64 v35; // rdi
  int v36; // edx
  __int64 v37; // rax
  __int64 v38; // r13
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  unsigned int v42; // [rsp+Ch] [rbp-124h]
  unsigned int v43; // [rsp+Ch] [rbp-124h]
  __int64 v44; // [rsp+10h] [rbp-120h] BYREF
  int v45; // [rsp+18h] [rbp-118h]
  __int64 v46; // [rsp+20h] [rbp-110h] BYREF
  int v47; // [rsp+28h] [rbp-108h]
  __m128i v48; // [rsp+30h] [rbp-100h] BYREF
  unsigned __int8 *v49; // [rsp+40h] [rbp-F0h]
  int v50; // [rsp+48h] [rbp-E8h]
  __m128i v51; // [rsp+50h] [rbp-E0h]
  unsigned __int8 *v52; // [rsp+60h] [rbp-D0h]
  int v53; // [rsp+68h] [rbp-C8h]
  __m128i v54; // [rsp+70h] [rbp-C0h]
  __m128i v55; // [rsp+80h] [rbp-B0h]
  __m128i v56; // [rsp+90h] [rbp-A0h]
  __m128i v57; // [rsp+A0h] [rbp-90h]
  __m128i v58; // [rsp+B0h] [rbp-80h]
  __m128i v59; // [rsp+C0h] [rbp-70h]
  __m128i v60; // [rsp+D0h] [rbp-60h]
  __m128i v61; // [rsp+E0h] [rbp-50h]
  __m128i v62; // [rsp+F0h] [rbp-40h]

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) <= 0x45u )
    sub_C64ED0("hmmastc is not supported on this architecture", 1u);
  v6 = *(_QWORD *)(a4 + 80);
  v44 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v44, v6, 1);
  v10 = *(_QWORD *)(a4 + 112);
  v45 = *(_DWORD *)(a4 + 72);
  v11 = sub_36D7800(v10);
  v12 = *(const __m128i **)(a4 + 40);
  v13 = v11;
  v14 = v12[10].m128i_i64[0];
  v15 = *(_DWORD *)(v14 + 24);
  if ( v15 != 35 && v15 != 11 )
    sub_C64ED0("rowcol not constant", 1u);
  v16 = *(_QWORD *)(v14 + 96);
  v17 = *(_QWORD **)(v16 + 24);
  if ( *(_DWORD *)(v16 + 32) > 0x40u )
    v17 = (_QWORD *)*v17;
  v18 = *(_QWORD *)(a4 + 80);
  if ( a2 )
  {
    v19 = _mm_loadu_si128(v12 + 5);
    v46 = *(_QWORD *)(a4 + 80);
    v48 = v19;
    if ( v18 )
    {
      v42 = v13;
      sub_B96E90((__int64)&v46, v18, 1);
      v13 = v42;
    }
    v20 = *(_QWORD *)(a1 + 64);
    v47 = *(_DWORD *)(a4 + 72);
    v49 = sub_3400BD0(v20, v13, (__int64)&v46, 7, 0, 1u, v19, 0);
    v50 = v21;
    if ( v46 )
      sub_B91220((__int64)&v46, v46);
    v22 = *(_QWORD *)(a4 + 80);
    v23 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 40) + 120LL));
    v46 = v22;
    v51 = v23;
    if ( v22 )
      sub_B96E90((__int64)&v46, v22, 1);
    v24 = *(_QWORD *)(a1 + 64);
    v47 = *(_DWORD *)(a4 + 72);
    v52 = sub_3400BD0(v24, (unsigned int)v17, (__int64)&v46, 7, 0, 1u, v19, 0);
    v53 = v26;
    if ( v46 )
      sub_B91220((__int64)&v46, v46);
    v27 = *(_QWORD *)(a4 + 40);
    v28 = *(_QWORD **)(a1 + 64);
    v29 = 13;
    v54 = _mm_loadu_si128((const __m128i *)(v27 + 200));
    v55 = _mm_loadu_si128((const __m128i *)(v27 + 240));
    v56 = _mm_loadu_si128((const __m128i *)(v27 + 280));
    v57 = _mm_loadu_si128((const __m128i *)(v27 + 320));
    v58 = _mm_loadu_si128((const __m128i *)(v27 + 360));
    v59 = _mm_loadu_si128((const __m128i *)(v27 + 400));
    v60 = _mm_loadu_si128((const __m128i *)(v27 + 440));
    v61 = _mm_loadu_si128((const __m128i *)(v27 + 480));
    v62 = _mm_loadu_si128((const __m128i *)v27);
  }
  else
  {
    v30 = _mm_loadu_si128(v12 + 5);
    v46 = *(_QWORD *)(a4 + 80);
    v48 = v30;
    if ( v18 )
    {
      v43 = v13;
      sub_B96E90((__int64)&v46, v18, 1);
      v13 = v43;
    }
    v31 = *(_QWORD *)(a1 + 64);
    v47 = *(_DWORD *)(a4 + 72);
    v49 = sub_3400BD0(v31, v13, (__int64)&v46, 7, 0, 1u, a5, 0);
    v50 = v32;
    if ( v46 )
      sub_B91220((__int64)&v46, v46);
    v33 = *(_QWORD *)(a4 + 80);
    v34 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 40) + 120LL));
    v46 = v33;
    v51 = v34;
    if ( v33 )
      sub_B96E90((__int64)&v46, v33, 1);
    v35 = *(_QWORD *)(a1 + 64);
    v47 = *(_DWORD *)(a4 + 72);
    v52 = sub_3400BD0(v35, (unsigned int)v17, (__int64)&v46, 7, 0, 1u, a5, 0);
    v53 = v36;
    if ( v46 )
      sub_B91220((__int64)&v46, v46);
    v37 = *(_QWORD *)(a4 + 40);
    v28 = *(_QWORD **)(a1 + 64);
    v29 = 9;
    v54 = _mm_loadu_si128((const __m128i *)(v37 + 200));
    v55 = _mm_loadu_si128((const __m128i *)(v37 + 240));
    v56 = _mm_loadu_si128((const __m128i *)(v37 + 280));
    v57 = _mm_loadu_si128((const __m128i *)(v37 + 320));
    v58 = _mm_loadu_si128((const __m128i *)v37);
  }
  v38 = sub_33E66D0(
          v28,
          a3,
          (__int64)&v44,
          *(_QWORD *)(a4 + 48),
          *(unsigned int *)(a4 + 68),
          v25,
          (unsigned __int64 *)&v48,
          v29);
  sub_34158F0(*(_QWORD *)(a1 + 64), a4, v38, v39, v40, v41);
  sub_3421DB0(v38);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a4);
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
}
