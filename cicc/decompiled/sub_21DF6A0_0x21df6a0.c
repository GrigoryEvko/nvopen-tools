// Function: sub_21DF6A0
// Address: 0x21df6a0
//
__int64 __fastcall sub_21DF6A0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v7; // rsi
  _QWORD *v8; // r14
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rcx
  _QWORD *v13; // rbx
  unsigned __int8 v14; // cl
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __int64 v17; // r9
  int v18; // edx
  __int64 v19; // r12
  __m128i v21; // xmm2
  __m128i v22; // xmm3
  int v23; // edx
  __int64 v24; // rsi
  __int64 v25; // rdi
  int v26; // edx
  __int64 v27; // rsi
  __int64 v28; // rdi
  __int64 v29; // r9
  int v30; // edx
  __int64 v31; // [rsp+8h] [rbp-A8h]
  __int64 v32; // [rsp+10h] [rbp-A0h] BYREF
  int v33; // [rsp+18h] [rbp-98h]
  __int64 v34; // [rsp+20h] [rbp-90h] BYREF
  int v35; // [rsp+28h] [rbp-88h]
  __m128i v36; // [rsp+30h] [rbp-80h] BYREF
  __m128i v37; // [rsp+40h] [rbp-70h]
  __int64 v38; // [rsp+50h] [rbp-60h]
  int v39; // [rsp+58h] [rbp-58h]
  __int64 v40; // [rsp+60h] [rbp-50h]
  int v41; // [rsp+68h] [rbp-48h]
  __int64 v42; // [rsp+70h] [rbp-40h]
  int v43; // [rsp+78h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 72);
  v8 = *(_QWORD **)(a1 - 176);
  v32 = v7;
  if ( v7 )
  {
    sub_1623A60((__int64)&v32, v7, 2);
    v7 = *(_QWORD *)(a2 + 72);
  }
  v9 = *(_DWORD *)(a2 + 64);
  v10 = *(_QWORD *)(a2 + 32);
  v33 = v9;
  v11 = *(_QWORD *)(*(_QWORD *)v10 + 88LL);
  if ( *(_DWORD *)(v11 + 32) <= 0x40u )
    v31 = *(_QWORD *)(v11 + 24);
  else
    v31 = **(_QWORD **)(v11 + 24);
  v12 = *(_QWORD *)(*(_QWORD *)(v10 + 40) + 88LL);
  v13 = *(_QWORD **)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v13 = (_QWORD *)*v13;
  v14 = (unsigned __int8)v13 & 7;
  if ( **(_BYTE **)(a2 + 40) == 9 )
  {
    v21 = _mm_loadu_si128((const __m128i *)(v10 + 80));
    v36 = v21;
    v22 = _mm_loadu_si128((const __m128i *)(v10 + 120));
    v34 = v7;
    v37 = v22;
    if ( v7 )
    {
      sub_1623A60((__int64)&v34, v7, 2);
      v9 = *(_DWORD *)(a2 + 64);
      v14 = (unsigned __int8)v13 & 7;
    }
    v35 = v9;
    v38 = sub_1D38BB0(*(_QWORD *)(a1 - 176), v14, (__int64)&v34, 5, 0, 1, a3, a4, v21, 0);
    v39 = v23;
    if ( v34 )
      sub_161E7C0((__int64)&v34, v34);
    v24 = *(_QWORD *)(a2 + 72);
    v34 = v24;
    if ( v24 )
      sub_1623A60((__int64)&v34, v24, 2);
    v25 = *(_QWORD *)(a1 - 176);
    v35 = *(_DWORD *)(a2 + 64);
    v40 = sub_1D38BB0(v25, (_DWORD)v31 == 4513, (__int64)&v34, 5, 0, 1, a3, a4, v21, 0);
    v41 = v26;
    if ( v34 )
      sub_161E7C0((__int64)&v34, v34);
    v27 = *(_QWORD *)(a2 + 72);
    v34 = v27;
    if ( v27 )
      sub_1623A60((__int64)&v34, v27, 2);
    v28 = *(_QWORD *)(a1 - 176);
    v35 = *(_DWORD *)(a2 + 64);
    v42 = sub_1D38BB0(v28, ((unsigned __int8)v13 & 8) != 0, (__int64)&v34, 5, 0, 1, a3, a4, v21, 0);
    v43 = v30;
    if ( v34 )
      sub_161E7C0((__int64)&v34, v34);
    v19 = sub_1D23DE0(v8, 3634, (__int64)&v32, *(_QWORD *)(a2 + 40), *(_DWORD *)(a2 + 60), v29, v36.m128i_i64, 5);
  }
  else
  {
    v15 = _mm_loadu_si128((const __m128i *)(v10 + 80));
    v36 = v15;
    v16 = _mm_loadu_si128((const __m128i *)(v10 + 120));
    v34 = v7;
    v37 = v16;
    if ( v7 )
    {
      sub_1623A60((__int64)&v34, v7, 2);
      v9 = *(_DWORD *)(a2 + 64);
      v14 = (unsigned __int8)v13 & 7;
    }
    v35 = v9;
    v38 = sub_1D38BB0(*(_QWORD *)(a1 - 176), v14, (__int64)&v34, 5, 0, 1, v15, *(double *)v16.m128i_i64, a5, 0);
    v39 = v18;
    if ( v34 )
      sub_161E7C0((__int64)&v34, v34);
    v19 = sub_1D23DE0(v8, 3635, (__int64)&v32, *(_QWORD *)(a2 + 40), *(_DWORD *)(a2 + 60), v17, v36.m128i_i64, 3);
  }
  if ( v32 )
    sub_161E7C0((__int64)&v32, v32);
  return v19;
}
