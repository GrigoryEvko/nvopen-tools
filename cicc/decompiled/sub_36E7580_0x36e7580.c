// Function: sub_36E7580
// Address: 0x36e7580
//
void __fastcall sub_36E7580(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v7; // rdi
  unsigned int v8; // eax
  const __m128i *v9; // rdx
  unsigned int v10; // r8d
  __int64 v11; // rcx
  int v12; // eax
  __int64 v13; // rax
  _QWORD *v14; // r14
  __m128i v15; // xmm0
  __int64 v16; // rsi
  __int64 v17; // rdi
  int v18; // edx
  __int64 v19; // rsi
  __m128i v20; // xmm1
  __int64 v21; // rdi
  __int64 v22; // r9
  int v23; // edx
  _QWORD *v24; // rdi
  unsigned __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r13
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned int v31; // [rsp+Ch] [rbp-A4h]
  __int64 v32; // [rsp+10h] [rbp-A0h] BYREF
  int v33; // [rsp+18h] [rbp-98h]
  __int64 v34; // [rsp+20h] [rbp-90h] BYREF
  int v35; // [rsp+28h] [rbp-88h]
  __m128i v36; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int8 *v37; // [rsp+40h] [rbp-70h]
  int v38; // [rsp+48h] [rbp-68h]
  __m128i v39; // [rsp+50h] [rbp-60h]
  unsigned __int8 *v40; // [rsp+60h] [rbp-50h]
  int v41; // [rsp+68h] [rbp-48h]
  __m128i v42; // [rsp+70h] [rbp-40h]

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) <= 0x45u )
    sub_C64ED0("hmmaldc is not supported on this architecture", 1u);
  v4 = *(_QWORD *)(a3 + 80);
  v32 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v32, v4, 1);
  v7 = *(_QWORD *)(a3 + 112);
  v33 = *(_DWORD *)(a3 + 72);
  v8 = sub_36D7800(v7);
  v9 = *(const __m128i **)(a3 + 40);
  v10 = v8;
  v11 = v9[10].m128i_i64[0];
  v12 = *(_DWORD *)(v11 + 24);
  if ( v12 != 35 && v12 != 11 )
    sub_C64ED0("rowcol not constant", 1u);
  v13 = *(_QWORD *)(v11 + 96);
  v14 = *(_QWORD **)(v13 + 24);
  if ( *(_DWORD *)(v13 + 32) > 0x40u )
    v14 = (_QWORD *)*v14;
  v15 = _mm_loadu_si128(v9 + 5);
  v16 = *(_QWORD *)(a3 + 80);
  v34 = v16;
  v36 = v15;
  if ( v16 )
  {
    v31 = v10;
    sub_B96E90((__int64)&v34, v16, 1);
    v10 = v31;
  }
  v17 = *(_QWORD *)(a1 + 64);
  v35 = *(_DWORD *)(a3 + 72);
  v37 = sub_3400BD0(v17, v10, (__int64)&v34, 7, 0, 1u, v15, 0);
  v38 = v18;
  if ( v34 )
    sub_B91220((__int64)&v34, v34);
  v19 = *(_QWORD *)(a3 + 80);
  v20 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a3 + 40) + 120LL));
  v34 = v19;
  v39 = v20;
  if ( v19 )
    sub_B96E90((__int64)&v34, v19, 1);
  v21 = *(_QWORD *)(a1 + 64);
  v35 = *(_DWORD *)(a3 + 72);
  v40 = sub_3400BD0(v21, (unsigned int)v14, (__int64)&v34, 7, 0, 1u, v15, 0);
  v41 = v23;
  if ( v34 )
    sub_B91220((__int64)&v34, v34);
  v24 = *(_QWORD **)(a1 + 64);
  v25 = *(_QWORD *)(a3 + 48);
  v26 = *(unsigned int *)(a3 + 68);
  v42 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a3 + 40));
  v27 = sub_33E66D0(v24, a2, (__int64)&v32, v25, v26, v22, (unsigned __int64 *)&v36, 5);
  sub_34158F0(*(_QWORD *)(a1 + 64), a3, v27, v28, v29, v30);
  sub_3421DB0(v27);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a3);
  if ( v32 )
    sub_B91220((__int64)&v32, v32);
}
