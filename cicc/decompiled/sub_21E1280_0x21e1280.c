// Function: sub_21E1280
// Address: 0x21e1280
//
__int64 __fastcall sub_21E1280(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        __int16 a4,
        __int64 a5,
        double a6,
        double a7,
        __m128i a8)
{
  unsigned int v8; // eax
  __int64 v13; // rsi
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rax
  unsigned int v17; // eax
  int v18; // r9d
  const __m128i *v19; // r8
  __m128i v20; // xmm0
  __int64 v21; // rsi
  unsigned int v22; // ecx
  __int64 v23; // rdi
  int v24; // edx
  __int64 v25; // rsi
  __m128i v26; // xmm1
  __int64 v27; // rdi
  int v28; // edx
  __int64 v29; // rsi
  __int64 v30; // rdi
  int v31; // edx
  __int64 v32; // rsi
  __int64 v33; // rdi
  __int64 v34; // r9
  int v35; // edx
  __int64 v36; // rcx
  int v37; // r8d
  __int64 v38; // r12
  __int64 v40; // [rsp+8h] [rbp-D8h]
  _QWORD *v41; // [rsp+10h] [rbp-D0h]
  unsigned int v42; // [rsp+18h] [rbp-C8h]
  __int64 v44; // [rsp+20h] [rbp-C0h] BYREF
  int v45; // [rsp+28h] [rbp-B8h]
  __int64 v46; // [rsp+30h] [rbp-B0h] BYREF
  int v47; // [rsp+38h] [rbp-A8h]
  __m128i v48; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v49; // [rsp+50h] [rbp-90h]
  int v50; // [rsp+58h] [rbp-88h]
  __m128i v51; // [rsp+60h] [rbp-80h]
  __int64 v52; // [rsp+70h] [rbp-70h]
  int v53; // [rsp+78h] [rbp-68h]
  __int64 v54; // [rsp+80h] [rbp-60h]
  int v55; // [rsp+88h] [rbp-58h]
  __int64 v56; // [rsp+90h] [rbp-50h]
  int v57; // [rsp+98h] [rbp-48h]
  __m128i v58; // [rsp+A0h] [rbp-40h]

  v8 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL);
  if ( v8 <= 0x47 || v8 == 72 && a3 > 1 )
    sub_16BD130("immaldab is not supported on this architecture", 1u);
  v13 = *(_QWORD *)(a5 + 72);
  v41 = *(_QWORD **)(a1 - 176);
  v44 = v13;
  if ( v13 )
    sub_1623A60((__int64)&v44, v13, 2);
  v14 = *(_QWORD *)(*(_QWORD *)(a5 + 32) + 160LL);
  v45 = *(_DWORD *)(a5 + 64);
  v15 = *(unsigned __int16 *)(v14 + 24);
  if ( v15 != 32 && v15 != 10 )
    sub_16BD130("rowcol not constant", 1u);
  v16 = *(_QWORD *)(v14 + 88);
  if ( *(_DWORD *)(v16 + 32) <= 0x40u )
    v40 = *(_QWORD *)(v16 + 24);
  else
    v40 = **(_QWORD **)(v16 + 24);
  v17 = sub_21DEF90(a5);
  v20 = _mm_loadu_si128(v19 + 5);
  v21 = *(_QWORD *)(a5 + 72);
  v22 = v17;
  v46 = v21;
  v48 = v20;
  if ( v21 )
  {
    v42 = v17;
    sub_1623A60((__int64)&v46, v21, 2);
    v18 = *(_DWORD *)(a5 + 64);
    v22 = v42;
  }
  v23 = *(_QWORD *)(a1 - 176);
  v47 = v18;
  v49 = sub_1D38BB0(v23, v22, (__int64)&v46, 5, 0, 1, v20, a7, a8, 0);
  v50 = v24;
  if ( v46 )
    sub_161E7C0((__int64)&v46, v46);
  v25 = *(_QWORD *)(a5 + 72);
  v26 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a5 + 32) + 120LL));
  v46 = v25;
  v51 = v26;
  if ( v25 )
    sub_1623A60((__int64)&v46, v25, 2);
  v27 = *(_QWORD *)(a1 - 176);
  v47 = *(_DWORD *)(a5 + 64);
  v52 = sub_1D38BB0(v27, a2, (__int64)&v46, 5, 0, 1, v20, *(double *)v26.m128i_i64, a8, 0);
  v53 = v28;
  if ( v46 )
    sub_161E7C0((__int64)&v46, v46);
  v29 = *(_QWORD *)(a5 + 72);
  v46 = v29;
  if ( v29 )
    sub_1623A60((__int64)&v46, v29, 2);
  v30 = *(_QWORD *)(a1 - 176);
  v47 = *(_DWORD *)(a5 + 64);
  v54 = sub_1D38BB0(v30, (unsigned int)v40, (__int64)&v46, 5, 0, 1, v20, *(double *)v26.m128i_i64, a8, 0);
  v55 = v31;
  if ( v46 )
    sub_161E7C0((__int64)&v46, v46);
  v32 = *(_QWORD *)(a5 + 72);
  v46 = v32;
  if ( v32 )
    sub_1623A60((__int64)&v46, v32, 2);
  v33 = *(_QWORD *)(a1 - 176);
  v47 = *(_DWORD *)(a5 + 64);
  v56 = sub_1D38BB0(v33, a3, (__int64)&v46, 5, 0, 1, v20, *(double *)v26.m128i_i64, a8, 0);
  v57 = v35;
  if ( v46 )
    sub_161E7C0((__int64)&v46, v46);
  v36 = *(_QWORD *)(a5 + 40);
  v37 = *(_DWORD *)(a5 + 60);
  v58 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a5 + 32));
  v38 = sub_1D23DE0(v41, a4, (__int64)&v44, v36, v37, v34, v48.m128i_i64, 7);
  if ( v44 )
    sub_161E7C0((__int64)&v44, v44);
  return v38;
}
