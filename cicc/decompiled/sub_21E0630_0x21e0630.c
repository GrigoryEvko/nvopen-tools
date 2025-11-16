// Function: sub_21E0630
// Address: 0x21e0630
//
__int64 __fastcall sub_21E0630(__int64 a1, __int16 a2, __int64 a3, double a4, double a5, __m128i a6)
{
  __int64 v7; // rsi
  _QWORD *v9; // r14
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rax
  unsigned int v13; // eax
  int v14; // r9d
  const __m128i *v15; // r8
  __m128i v16; // xmm0
  __int64 v17; // rsi
  unsigned int v18; // ecx
  int v19; // edx
  __int64 v20; // rsi
  __m128i v21; // xmm1
  __int64 v22; // rdi
  __int64 v23; // r9
  int v24; // edx
  __int64 v25; // rcx
  int v26; // r8d
  __int64 v27; // r12
  unsigned int v29; // [rsp+4h] [rbp-ACh]
  __int64 v30; // [rsp+8h] [rbp-A8h]
  __int64 v31; // [rsp+10h] [rbp-A0h] BYREF
  int v32; // [rsp+18h] [rbp-98h]
  __int64 v33; // [rsp+20h] [rbp-90h] BYREF
  int v34; // [rsp+28h] [rbp-88h]
  __m128i v35; // [rsp+30h] [rbp-80h] BYREF
  __int64 v36; // [rsp+40h] [rbp-70h]
  int v37; // [rsp+48h] [rbp-68h]
  __m128i v38; // [rsp+50h] [rbp-60h]
  __int64 v39; // [rsp+60h] [rbp-50h]
  int v40; // [rsp+68h] [rbp-48h]
  __m128i v41; // [rsp+70h] [rbp-40h]

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL) <= 0x45u )
    sub_16BD130("hmmaldc is not supported on this architecture", 1u);
  v7 = *(_QWORD *)(a3 + 72);
  v9 = *(_QWORD **)(a1 - 176);
  v31 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v31, v7, 2);
  v10 = *(_QWORD *)(*(_QWORD *)(a3 + 32) + 160LL);
  v32 = *(_DWORD *)(a3 + 64);
  v11 = *(unsigned __int16 *)(v10 + 24);
  if ( v11 != 32 && v11 != 10 )
    sub_16BD130("rowcol not constant", 1u);
  v12 = *(_QWORD *)(v10 + 88);
  if ( *(_DWORD *)(v12 + 32) <= 0x40u )
    v30 = *(_QWORD *)(v12 + 24);
  else
    v30 = **(_QWORD **)(v12 + 24);
  v13 = sub_21DEF90(a3);
  v16 = _mm_loadu_si128(v15 + 5);
  v17 = *(_QWORD *)(a3 + 72);
  v18 = v13;
  v33 = v17;
  v35 = v16;
  if ( v17 )
  {
    v29 = v13;
    sub_1623A60((__int64)&v33, v17, 2);
    v14 = *(_DWORD *)(a3 + 64);
    v18 = v29;
  }
  v34 = v14;
  v36 = sub_1D38BB0(*(_QWORD *)(a1 - 176), v18, (__int64)&v33, 5, 0, 1, v16, a5, a6, 0);
  v37 = v19;
  if ( v33 )
    sub_161E7C0((__int64)&v33, v33);
  v20 = *(_QWORD *)(a3 + 72);
  v21 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a3 + 32) + 120LL));
  v33 = v20;
  v38 = v21;
  if ( v20 )
    sub_1623A60((__int64)&v33, v20, 2);
  v22 = *(_QWORD *)(a1 - 176);
  v34 = *(_DWORD *)(a3 + 64);
  v39 = sub_1D38BB0(v22, (unsigned int)v30, (__int64)&v33, 5, 0, 1, v16, *(double *)v21.m128i_i64, a6, 0);
  v40 = v24;
  if ( v33 )
    sub_161E7C0((__int64)&v33, v33);
  v25 = *(_QWORD *)(a3 + 40);
  v26 = *(_DWORD *)(a3 + 60);
  v41 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a3 + 32));
  v27 = sub_1D23DE0(v9, a2, (__int64)&v31, v25, v26, v23, v35.m128i_i64, 5);
  if ( v31 )
    sub_161E7C0((__int64)&v31, v31);
  return v27;
}
