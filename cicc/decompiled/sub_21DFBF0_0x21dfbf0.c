// Function: sub_21DFBF0
// Address: 0x21dfbf0
//
__int64 __fastcall sub_21DFBF0(__int64 a1, char a2, __int16 a3, __int64 a4, __m128i a5, double a6, __m128i a7)
{
  _QWORD *v7; // r15
  __int64 v9; // rsi
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rax
  _QWORD *v14; // r14
  unsigned int v15; // eax
  int v16; // r8d
  const __m128i *v17; // r9
  __int64 v18; // rsi
  unsigned int v19; // ecx
  __m128i v20; // xmm0
  int v21; // edx
  __int64 v22; // rsi
  __m128i v23; // xmm1
  __int64 v24; // rdi
  __int64 v25; // r9
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rdx
  __m128i v29; // xmm3
  int v30; // edx
  __int64 v31; // rsi
  __m128i v32; // xmm4
  __int64 v33; // rdi
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // r12
  unsigned int v38; // [rsp+8h] [rbp-128h]
  unsigned int v39; // [rsp+8h] [rbp-128h]
  __int64 v41; // [rsp+10h] [rbp-120h] BYREF
  int v42; // [rsp+18h] [rbp-118h]
  __int64 v43; // [rsp+20h] [rbp-110h] BYREF
  int v44; // [rsp+28h] [rbp-108h]
  __m128i v45; // [rsp+30h] [rbp-100h] BYREF
  __int64 v46; // [rsp+40h] [rbp-F0h]
  int v47; // [rsp+48h] [rbp-E8h]
  __m128i v48; // [rsp+50h] [rbp-E0h]
  __int64 v49; // [rsp+60h] [rbp-D0h]
  int v50; // [rsp+68h] [rbp-C8h]
  __m128i v51; // [rsp+70h] [rbp-C0h]
  __m128i v52; // [rsp+80h] [rbp-B0h]
  __m128i v53; // [rsp+90h] [rbp-A0h]
  __m128i v54; // [rsp+A0h] [rbp-90h]
  __m128i v55; // [rsp+B0h] [rbp-80h]
  __m128i v56; // [rsp+C0h] [rbp-70h]
  __m128i v57; // [rsp+D0h] [rbp-60h]
  __m128i v58; // [rsp+E0h] [rbp-50h]
  __m128i v59; // [rsp+F0h] [rbp-40h]

  v7 = *(_QWORD **)(a1 - 176);
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL) <= 0x45u )
    sub_16BD130("hmmastc is not supported on this architecture", 1u);
  v9 = *(_QWORD *)(a4 + 72);
  v41 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v41, v9, 2);
  v11 = *(_QWORD *)(*(_QWORD *)(a4 + 32) + 160LL);
  v42 = *(_DWORD *)(a4 + 64);
  v12 = *(unsigned __int16 *)(v11 + 24);
  if ( v12 != 32 && v12 != 10 )
    sub_16BD130("rowcol not constant", 1u);
  v13 = *(_QWORD *)(v11 + 88);
  v14 = *(_QWORD **)(v13 + 24);
  if ( *(_DWORD *)(v13 + 32) > 0x40u )
    v14 = (_QWORD *)*v14;
  v15 = sub_21DEF90(a4);
  v18 = *(_QWORD *)(a4 + 72);
  v19 = v15;
  if ( a2 )
  {
    v20 = _mm_loadu_si128(v17 + 5);
    v43 = *(_QWORD *)(a4 + 72);
    v45 = v20;
    if ( v18 )
    {
      v38 = v15;
      sub_1623A60((__int64)&v43, v18, 2);
      v16 = *(_DWORD *)(a4 + 64);
      v19 = v38;
    }
    v44 = v16;
    v46 = sub_1D38BB0(*(_QWORD *)(a1 - 176), v19, (__int64)&v43, 5, 0, 1, v20, a6, a7, 0);
    v47 = v21;
    if ( v43 )
      sub_161E7C0((__int64)&v43, v43);
    v22 = *(_QWORD *)(a4 + 72);
    v23 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 32) + 120LL));
    v43 = v22;
    v48 = v23;
    if ( v22 )
      sub_1623A60((__int64)&v43, v22, 2);
    v24 = *(_QWORD *)(a1 - 176);
    v44 = *(_DWORD *)(a4 + 64);
    v49 = sub_1D38BB0(v24, (unsigned int)v14, (__int64)&v43, 5, 0, 1, v20, *(double *)v23.m128i_i64, a7, 0);
    v50 = v26;
    if ( v43 )
      sub_161E7C0((__int64)&v43, v43);
    v27 = *(_QWORD *)(a4 + 32);
    v28 = 13;
    v51 = _mm_loadu_si128((const __m128i *)(v27 + 200));
    v52 = _mm_loadu_si128((const __m128i *)(v27 + 240));
    v53 = _mm_loadu_si128((const __m128i *)(v27 + 280));
    v54 = _mm_loadu_si128((const __m128i *)(v27 + 320));
    v55 = _mm_loadu_si128((const __m128i *)(v27 + 360));
    v56 = _mm_loadu_si128((const __m128i *)(v27 + 400));
    v57 = _mm_loadu_si128((const __m128i *)(v27 + 440));
    v58 = _mm_loadu_si128((const __m128i *)(v27 + 480));
    v59 = _mm_loadu_si128((const __m128i *)v27);
  }
  else
  {
    v29 = _mm_loadu_si128(v17 + 5);
    v43 = *(_QWORD *)(a4 + 72);
    v45 = v29;
    if ( v18 )
    {
      v39 = v15;
      sub_1623A60((__int64)&v43, v18, 2);
      v16 = *(_DWORD *)(a4 + 64);
      v19 = v39;
    }
    v44 = v16;
    v46 = sub_1D38BB0(*(_QWORD *)(a1 - 176), v19, (__int64)&v43, 5, 0, 1, a5, a6, a7, 0);
    v47 = v30;
    if ( v43 )
      sub_161E7C0((__int64)&v43, v43);
    v31 = *(_QWORD *)(a4 + 72);
    v32 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 32) + 120LL));
    v43 = v31;
    v48 = v32;
    if ( v31 )
      sub_1623A60((__int64)&v43, v31, 2);
    v33 = *(_QWORD *)(a1 - 176);
    v44 = *(_DWORD *)(a4 + 64);
    v49 = sub_1D38BB0(v33, (unsigned int)v14, (__int64)&v43, 5, 0, 1, a5, a6, a7, 0);
    v50 = v34;
    if ( v43 )
      sub_161E7C0((__int64)&v43, v43);
    v35 = *(_QWORD *)(a4 + 32);
    v28 = 9;
    v51 = _mm_loadu_si128((const __m128i *)(v35 + 200));
    v52 = _mm_loadu_si128((const __m128i *)(v35 + 240));
    v53 = _mm_loadu_si128((const __m128i *)(v35 + 280));
    v54 = _mm_loadu_si128((const __m128i *)(v35 + 320));
    v55 = _mm_loadu_si128((const __m128i *)v35);
  }
  v36 = sub_1D23DE0(v7, a3, (__int64)&v41, *(_QWORD *)(a4 + 40), *(_DWORD *)(a4 + 60), v25, v45.m128i_i64, v28);
  if ( v41 )
    sub_161E7C0((__int64)&v41, v41);
  return v36;
}
