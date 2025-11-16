// Function: sub_211B770
// Address: 0x211b770
//
__int64 *__fastcall sub_211B770(__int64 *a1, unsigned __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  __int128 v8; // xmm0
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // r11
  unsigned int v13; // r10d
  __int64 v14; // rdx
  __int64 *v15; // rdi
  __int64 v16; // rcx
  _QWORD *v17; // rax
  __int64 *v18; // rax
  const __m128i *v19; // r9
  __int64 *v20; // r15
  unsigned int v22; // [rsp+0h] [rbp-70h]
  __int64 v23; // [rsp+8h] [rbp-68h]
  __int64 v24; // [rsp+20h] [rbp-50h] BYREF
  int v25; // [rsp+28h] [rbp-48h]
  __int64 v26; // [rsp+30h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 40));
  v9 = *(_QWORD *)v7;
  v10 = *(_QWORD *)(v7 + 8);
  sub_1F40D10(
    (__int64)&v24,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v11 = *(_QWORD *)(a2 + 72);
  v12 = v26;
  v13 = (unsigned __int8)v25;
  v24 = v11;
  if ( v11 )
  {
    v22 = (unsigned __int8)v25;
    v23 = v26;
    sub_1623A60((__int64)&v24, v11, 2);
    v13 = v22;
    v12 = v23;
  }
  v14 = *(_QWORD *)(a2 + 32);
  v15 = (__int64 *)a1[1];
  v25 = *(_DWORD *)(a2 + 64);
  v16 = *(_QWORD *)(*(_QWORD *)(v14 + 120) + 88LL);
  v17 = *(_QWORD **)(v16 + 24);
  if ( *(_DWORD *)(v16 + 32) > 0x40u )
    v17 = (_QWORD *)*v17;
  v18 = sub_1D38F20(
          v15,
          v13,
          v12,
          (__int64)&v24,
          v9,
          v10,
          *(double *)&v8,
          a4,
          a5,
          v8,
          *(_OWORD *)(v14 + 80),
          (unsigned int)v17);
  v20 = v18;
  if ( (__int64 *)a2 != v18 )
    sub_2013400((__int64)a1, a2, 1, (__int64)v18, (__m128i *)1, v19);
  if ( v24 )
    sub_161E7C0((__int64)&v24, v24);
  return v20;
}
