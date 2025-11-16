// Function: sub_21E3600
// Address: 0x21e3600
//
__int64 __fastcall sub_21E3600(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  int v8; // edi
  __int64 v9; // rdx
  __int64 v10; // rax
  bool v11; // cc
  _QWORD *v12; // rax
  __int64 v13; // r8
  _QWORD *v14; // rcx
  unsigned __int8 v15; // cl
  int v16; // r12d
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  __int32 v19; // edx
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // r9
  int v23; // edx
  __int64 v24; // r12
  __int64 v26; // r12
  __m128i v27; // xmm2
  __m128i v28; // xmm4
  int v29; // edx
  __int64 v30; // rsi
  __int64 v31; // rdi
  __int64 v32; // r9
  int v33; // edx
  unsigned __int8 v34; // [rsp+Fh] [rbp-B1h]
  _QWORD *v35; // [rsp+10h] [rbp-B0h]
  __int64 v36; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v37; // [rsp+18h] [rbp-A8h]
  __int64 v38; // [rsp+20h] [rbp-A0h] BYREF
  int v39; // [rsp+28h] [rbp-98h]
  __int64 v40; // [rsp+30h] [rbp-90h] BYREF
  int v41; // [rsp+38h] [rbp-88h]
  __m128i v42; // [rsp+40h] [rbp-80h] BYREF
  __m128i v43; // [rsp+50h] [rbp-70h]
  __m128i v44; // [rsp+60h] [rbp-60h]
  __int64 v45; // [rsp+70h] [rbp-50h]
  int v46; // [rsp+78h] [rbp-48h]
  __int64 v47; // [rsp+80h] [rbp-40h]
  int v48; // [rsp+88h] [rbp-38h]

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL) <= 0x63u )
    return 0;
  v6 = *(_QWORD *)(a2 + 72);
  v35 = *(_QWORD **)(a1 - 176);
  v38 = v6;
  if ( v6 )
  {
    sub_1623A60((__int64)&v38, v6, 2);
    v6 = *(_QWORD *)(a2 + 72);
  }
  v8 = *(_DWORD *)(a2 + 64);
  v9 = *(_QWORD *)(a2 + 32);
  v39 = v8;
  v10 = *(_QWORD *)(*(_QWORD *)v9 + 88LL);
  v11 = *(_DWORD *)(v10 + 32) <= 0x40u;
  v12 = *(_QWORD **)(v10 + 24);
  if ( !v11 )
    v12 = (_QWORD *)*v12;
  v13 = *(_QWORD *)(*(_QWORD *)(v9 + 40) + 88LL);
  v14 = *(_QWORD **)(v13 + 24);
  if ( *(_DWORD *)(v13 + 32) > 0x40u )
    v14 = (_QWORD *)*v14;
  v15 = (unsigned __int8)v14 & 7;
  switch ( (_DWORD)v12 )
  {
    case 0xF59:
      v26 = 1;
      goto LABEL_30;
    case 0x105E:
      v36 = 1;
      LOWORD(v16) = 3218;
      goto LABEL_17;
    case 0xE3A:
      v36 = 1;
      goto LABEL_14;
  }
  if ( (unsigned int)v12 > 0xF5A )
  {
    LOWORD(v16) = 0;
    v36 = 0;
    if ( (_DWORD)v12 == 4193 )
      LOWORD(v16) = 3218;
    goto LABEL_17;
  }
  if ( (unsigned int)v12 > 0xF58 )
  {
    v26 = 0;
LABEL_30:
    v27 = _mm_loadu_si128((const __m128i *)(v9 + 80));
    v42 = v27;
    v43 = _mm_loadu_si128((const __m128i *)(v9 + 120));
    v28 = _mm_loadu_si128((const __m128i *)(v9 + 160));
    v40 = v6;
    v44 = v28;
    if ( v6 )
    {
      v37 = v15;
      sub_1623A60((__int64)&v40, v6, 2);
      v8 = *(_DWORD *)(a2 + 64);
      v15 = v37;
    }
    v41 = v8;
    v45 = sub_1D38BB0(*(_QWORD *)(a1 - 176), v15, (__int64)&v40, 5, 0, 1, a3, a4, v27, 0);
    v46 = v29;
    if ( v40 )
      sub_161E7C0((__int64)&v40, v40);
    v30 = *(_QWORD *)(a2 + 72);
    v40 = v30;
    if ( v30 )
      sub_1623A60((__int64)&v40, v30, 2);
    v31 = *(_QWORD *)(a1 - 176);
    v41 = *(_DWORD *)(a2 + 64);
    v47 = sub_1D38BB0(v31, v26, (__int64)&v40, 5, 0, 1, a3, a4, v27, 0);
    v48 = v33;
    if ( v40 )
      sub_161E7C0((__int64)&v40, v40);
    v24 = sub_1D23DE0(v35, 470, (__int64)&v38, *(_QWORD *)(a2 + 40), *(_DWORD *)(a2 + 60), v32, v42.m128i_i64, 5);
    goto LABEL_26;
  }
  v36 = 0;
LABEL_14:
  v16 = (unsigned int)((_DWORD)v12 - 3642) < 2 ? 0x86 : 0;
LABEL_17:
  v17 = _mm_loadu_si128((const __m128i *)(v9 + 80));
  v42 = v17;
  v18 = _mm_loadu_si128((const __m128i *)(v9 + 120));
  v40 = v6;
  v43 = v18;
  if ( v6 )
  {
    v34 = v15;
    sub_1623A60((__int64)&v40, v6, 2);
    v8 = *(_DWORD *)(a2 + 64);
    v15 = v34;
  }
  v41 = v8;
  v44.m128i_i64[0] = sub_1D38BB0(
                       *(_QWORD *)(a1 - 176),
                       v15,
                       (__int64)&v40,
                       5,
                       0,
                       1,
                       v17,
                       *(double *)v18.m128i_i64,
                       a5,
                       0);
  v44.m128i_i32[2] = v19;
  if ( v40 )
    sub_161E7C0((__int64)&v40, v40);
  v20 = *(_QWORD *)(a2 + 72);
  v40 = v20;
  if ( v20 )
    sub_1623A60((__int64)&v40, v20, 2);
  v21 = *(_QWORD *)(a1 - 176);
  v41 = *(_DWORD *)(a2 + 64);
  v45 = sub_1D38BB0(v21, v36, (__int64)&v40, 5, 0, 1, v17, *(double *)v18.m128i_i64, a5, 0);
  v46 = v23;
  if ( v40 )
    sub_161E7C0((__int64)&v40, v40);
  v24 = sub_1D23DE0(v35, v16, (__int64)&v38, *(_QWORD *)(a2 + 40), *(_DWORD *)(a2 + 60), v22, v42.m128i_i64, 4);
LABEL_26:
  if ( v38 )
    sub_161E7C0((__int64)&v38, v38);
  return v24;
}
