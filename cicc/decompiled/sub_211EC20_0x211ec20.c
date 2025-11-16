// Function: sub_211EC20
// Address: 0x211ec20
//
__int64 *__fastcall sub_211EC20(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __m128 v8; // xmm0
  __m128i v9; // xmm1
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned int v15; // esi
  _QWORD *v16; // r13
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v21; // rsi
  __int64 v22; // r14
  const void **v23; // r8
  unsigned int v24; // r15d
  int v25; // edx
  __int64 v26; // [rsp-8h] [rbp-98h]
  const void **v27; // [rsp+8h] [rbp-88h]
  unsigned int v28; // [rsp+2Ch] [rbp-64h] BYREF
  __int128 v29; // [rsp+30h] [rbp-60h] BYREF
  __int128 v30; // [rsp+40h] [rbp-50h] BYREF
  __int64 v31; // [rsp+50h] [rbp-40h] BYREF
  int v32; // [rsp+58h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = (__m128)_mm_loadu_si128((const __m128i *)(v6 + 80));
  v9 = _mm_loadu_si128((const __m128i *)(v6 + 120));
  v10 = *(_QWORD *)(v6 + 40);
  v29 = (__int128)v8;
  LODWORD(v10) = *(_DWORD *)(v10 + 84);
  v30 = (__int128)v9;
  v31 = v7;
  v28 = v10;
  if ( v7 )
    sub_1623A60((__int64)&v31, v7, 2);
  v32 = *(_DWORD *)(a2 + 64);
  sub_211E5A0(
    (__int64 **)a1,
    (unsigned __int64 *)&v29,
    (__int64)&v30,
    &v28,
    (__int64)&v31,
    v8,
    *(double *)v9.m128i_i64,
    a5);
  if ( v31 )
    sub_161E7C0((__int64)&v31, v31);
  if ( (_QWORD)v30 )
  {
    v15 = v28;
  }
  else
  {
    v21 = *(_QWORD *)(a2 + 72);
    v22 = *(_QWORD *)(a1 + 8);
    v23 = *(const void ***)(*(_QWORD *)(v29 + 40) + 16LL * DWORD2(v29) + 8);
    v24 = *(unsigned __int8 *)(*(_QWORD *)(v29 + 40) + 16LL * DWORD2(v29));
    v31 = v21;
    if ( v21 )
    {
      v27 = v23;
      sub_1623A60((__int64)&v31, v21, 2);
      v23 = v27;
    }
    v32 = *(_DWORD *)(a2 + 64);
    *(_QWORD *)&v30 = sub_1D38BB0(v22, 0, (__int64)&v31, v24, v23, 0, (__m128i)v8, *(double *)v9.m128i_i64, a5, 0);
    DWORD2(v30) = v25;
    v11 = v26;
    if ( v31 )
      sub_161E7C0((__int64)&v31, v31);
    v28 = 22;
    v15 = 22;
  }
  v16 = *(_QWORD **)(a1 + 8);
  v17 = *(_QWORD *)(a2 + 32);
  v18 = sub_1D28D50(v16, v15, v11, v12, v13, v14);
  return sub_1D2E370(
           v16,
           (__int64 *)a2,
           **(_QWORD **)(a2 + 32),
           *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
           v18,
           v19,
           v29,
           v30,
           *(_OWORD *)(v17 + 160));
}
