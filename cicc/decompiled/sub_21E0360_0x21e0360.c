// Function: sub_21E0360
// Address: 0x21e0360
//
__int64 __fastcall sub_21E0360(__int64 a1, unsigned int a2, __int16 a3, __int64 a4, double a5, double a6, __m128i a7)
{
  __int64 v7; // rsi
  _QWORD *v11; // r15
  __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rax
  unsigned int v15; // eax
  int v16; // r9d
  const __m128i *v17; // rcx
  __m128i v18; // xmm0
  __int64 v19; // rsi
  unsigned int v20; // r8d
  __int64 v21; // rdi
  int v22; // edx
  __int64 v23; // rsi
  __m128i v24; // xmm1
  __int64 v25; // rdi
  int v26; // edx
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // r9
  int v30; // edx
  __int64 v31; // rcx
  int v32; // r8d
  __int64 v33; // r12
  __int64 v35; // [rsp+0h] [rbp-C0h]
  unsigned int v36; // [rsp+8h] [rbp-B8h]
  __int64 v38; // [rsp+10h] [rbp-B0h] BYREF
  int v39; // [rsp+18h] [rbp-A8h]
  __int64 v40; // [rsp+20h] [rbp-A0h] BYREF
  int v41; // [rsp+28h] [rbp-98h]
  __m128i v42; // [rsp+30h] [rbp-90h] BYREF
  __int64 v43; // [rsp+40h] [rbp-80h]
  int v44; // [rsp+48h] [rbp-78h]
  __m128i v45; // [rsp+50h] [rbp-70h]
  __int64 v46; // [rsp+60h] [rbp-60h]
  int v47; // [rsp+68h] [rbp-58h]
  __int64 v48; // [rsp+70h] [rbp-50h]
  int v49; // [rsp+78h] [rbp-48h]
  __m128i v50; // [rsp+80h] [rbp-40h]

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL) <= 0x45u )
    sub_16BD130("hmmaldab is not supported on this architecture", 1u);
  v7 = *(_QWORD *)(a4 + 72);
  v11 = *(_QWORD **)(a1 - 176);
  v38 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v38, v7, 2);
  v12 = *(_QWORD *)(*(_QWORD *)(a4 + 32) + 160LL);
  v39 = *(_DWORD *)(a4 + 64);
  v13 = *(unsigned __int16 *)(v12 + 24);
  if ( v13 != 32 && v13 != 10 )
    sub_16BD130("rowcol not constant", 1u);
  v14 = *(_QWORD *)(v12 + 88);
  if ( *(_DWORD *)(v14 + 32) <= 0x40u )
    v35 = *(_QWORD *)(v14 + 24);
  else
    v35 = **(_QWORD **)(v14 + 24);
  v15 = sub_21DEF90(a4);
  v18 = _mm_loadu_si128(v17 + 5);
  v19 = *(_QWORD *)(a4 + 72);
  v20 = v15;
  v40 = v19;
  v42 = v18;
  if ( v19 )
  {
    v36 = v15;
    sub_1623A60((__int64)&v40, v19, 2);
    v16 = *(_DWORD *)(a4 + 64);
    v20 = v36;
  }
  v21 = *(_QWORD *)(a1 - 176);
  v41 = v16;
  v43 = sub_1D38BB0(v21, v20, (__int64)&v40, 5, 0, 1, v18, a6, a7, 0);
  v44 = v22;
  if ( v40 )
    sub_161E7C0((__int64)&v40, v40);
  v23 = *(_QWORD *)(a4 + 72);
  v24 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 32) + 120LL));
  v40 = v23;
  v45 = v24;
  if ( v23 )
    sub_1623A60((__int64)&v40, v23, 2);
  v25 = *(_QWORD *)(a1 - 176);
  v41 = *(_DWORD *)(a4 + 64);
  v46 = sub_1D38BB0(v25, a2, (__int64)&v40, 5, 0, 1, v18, *(double *)v24.m128i_i64, a7, 0);
  v47 = v26;
  if ( v40 )
    sub_161E7C0((__int64)&v40, v40);
  v27 = *(_QWORD *)(a4 + 72);
  v40 = v27;
  if ( v27 )
    sub_1623A60((__int64)&v40, v27, 2);
  v28 = *(_QWORD *)(a1 - 176);
  v41 = *(_DWORD *)(a4 + 64);
  v48 = sub_1D38BB0(v28, (unsigned int)v35, (__int64)&v40, 5, 0, 1, v18, *(double *)v24.m128i_i64, a7, 0);
  v49 = v30;
  if ( v40 )
    sub_161E7C0((__int64)&v40, v40);
  v31 = *(_QWORD *)(a4 + 40);
  v32 = *(_DWORD *)(a4 + 60);
  v50 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a4 + 32));
  v33 = sub_1D23DE0(v11, a3, (__int64)&v38, v31, v32, v29, v42.m128i_i64, 6);
  if ( v38 )
    sub_161E7C0((__int64)&v38, v38);
  return v33;
}
