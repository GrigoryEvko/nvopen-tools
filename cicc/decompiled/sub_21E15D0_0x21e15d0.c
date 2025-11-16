// Function: sub_21E15D0
// Address: 0x21e15d0
//
__int64 __fastcall sub_21E15D0(__int64 a1, int a2, __int64 a3, double a4, double a5, __m128i a6)
{
  unsigned int v6; // eax
  __int16 v8; // r13
  __int64 v9; // rsi
  _QWORD *v10; // r14
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rax
  unsigned int v14; // eax
  int v15; // r9d
  const __m128i *v16; // r8
  __m128i v17; // xmm0
  __int64 v18; // rsi
  unsigned int v19; // ecx
  int v20; // edx
  __int64 v21; // rsi
  __m128i v22; // xmm1
  __int64 v23; // rdi
  __int64 v24; // r9
  int v25; // edx
  __int64 v26; // rcx
  int v27; // r8d
  __int64 v28; // r12
  unsigned int v30; // [rsp+4h] [rbp-ACh]
  __int64 v31; // [rsp+8h] [rbp-A8h]
  __int64 v32; // [rsp+10h] [rbp-A0h] BYREF
  int v33; // [rsp+18h] [rbp-98h]
  __int64 v34; // [rsp+20h] [rbp-90h] BYREF
  int v35; // [rsp+28h] [rbp-88h]
  __m128i v36; // [rsp+30h] [rbp-80h] BYREF
  __int64 v37; // [rsp+40h] [rbp-70h]
  int v38; // [rsp+48h] [rbp-68h]
  __m128i v39; // [rsp+50h] [rbp-60h]
  __int64 v40; // [rsp+60h] [rbp-50h]
  int v41; // [rsp+68h] [rbp-48h]
  __m128i v42; // [rsp+70h] [rbp-40h]

  v6 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL);
  if ( v6 <= 0x47 || ((v8 = a2, (unsigned int)(a2 - 610) <= 1) || (unsigned int)(a2 - 179) <= 1) && v6 == 72 )
    sub_16BD130("immaldc is not supported on this architecture", 1u);
  v9 = *(_QWORD *)(a3 + 72);
  v10 = *(_QWORD **)(a1 - 176);
  v32 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v32, v9, 2);
  v11 = *(_QWORD *)(*(_QWORD *)(a3 + 32) + 160LL);
  v33 = *(_DWORD *)(a3 + 64);
  v12 = *(unsigned __int16 *)(v11 + 24);
  if ( v12 != 32 && v12 != 10 )
    sub_16BD130("rowcol not constant", 1u);
  v13 = *(_QWORD *)(v11 + 88);
  if ( *(_DWORD *)(v13 + 32) <= 0x40u )
    v31 = *(_QWORD *)(v13 + 24);
  else
    v31 = **(_QWORD **)(v13 + 24);
  v14 = sub_21DEF90(a3);
  v17 = _mm_loadu_si128(v16 + 5);
  v18 = *(_QWORD *)(a3 + 72);
  v19 = v14;
  v34 = v18;
  v36 = v17;
  if ( v18 )
  {
    v30 = v14;
    sub_1623A60((__int64)&v34, v18, 2);
    v15 = *(_DWORD *)(a3 + 64);
    v19 = v30;
  }
  v35 = v15;
  v37 = sub_1D38BB0(*(_QWORD *)(a1 - 176), v19, (__int64)&v34, 5, 0, 1, v17, a5, a6, 0);
  v38 = v20;
  if ( v34 )
    sub_161E7C0((__int64)&v34, v34);
  v21 = *(_QWORD *)(a3 + 72);
  v22 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a3 + 32) + 120LL));
  v34 = v21;
  v39 = v22;
  if ( v21 )
    sub_1623A60((__int64)&v34, v21, 2);
  v23 = *(_QWORD *)(a1 - 176);
  v35 = *(_DWORD *)(a3 + 64);
  v40 = sub_1D38BB0(v23, (unsigned int)v31, (__int64)&v34, 5, 0, 1, v17, *(double *)v22.m128i_i64, a6, 0);
  v41 = v25;
  if ( v34 )
    sub_161E7C0((__int64)&v34, v34);
  v26 = *(_QWORD *)(a3 + 40);
  v27 = *(_DWORD *)(a3 + 60);
  v42 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a3 + 32));
  v28 = sub_1D23DE0(v10, v8, (__int64)&v32, v26, v27, v24, v36.m128i_i64, 5);
  if ( v32 )
    sub_161E7C0((__int64)&v32, v32);
  return v28;
}
