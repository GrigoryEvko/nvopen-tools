// Function: sub_36E7EA0
// Address: 0x36e7ea0
//
void __fastcall sub_36E7EA0(__int64 a1, int a2, __int64 a3)
{
  unsigned int v3; // eax
  __int64 v7; // rsi
  __int64 v8; // rdi
  unsigned int v9; // eax
  const __m128i *v10; // rdx
  unsigned int v11; // r8d
  __int64 v12; // rcx
  int v13; // eax
  __int64 v14; // rax
  _QWORD *v15; // r14
  __m128i v16; // xmm0
  __int64 v17; // rsi
  __int64 v18; // rdi
  int v19; // edx
  __int64 v20; // rsi
  __m128i v21; // xmm1
  __int64 v22; // rdi
  __int64 v23; // r9
  int v24; // edx
  _QWORD *v25; // rdi
  unsigned __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r13
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned int v32; // [rsp+Ch] [rbp-A4h]
  __int64 v33; // [rsp+10h] [rbp-A0h] BYREF
  int v34; // [rsp+18h] [rbp-98h]
  __int64 v35; // [rsp+20h] [rbp-90h] BYREF
  int v36; // [rsp+28h] [rbp-88h]
  __m128i v37; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int8 *v38; // [rsp+40h] [rbp-70h]
  int v39; // [rsp+48h] [rbp-68h]
  __m128i v40; // [rsp+50h] [rbp-60h]
  unsigned __int8 *v41; // [rsp+60h] [rbp-50h]
  int v42; // [rsp+68h] [rbp-48h]
  __m128i v43; // [rsp+70h] [rbp-40h]

  v3 = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL);
  if ( v3 <= 0x47 || ((unsigned int)(a2 - 1596) <= 1 || (unsigned int)(a2 - 387) <= 1) && v3 != 72 )
    sub_C64ED0("immaldc is not supported on this architecture", 1u);
  v7 = *(_QWORD *)(a3 + 80);
  v33 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v33, v7, 1);
  v8 = *(_QWORD *)(a3 + 112);
  v34 = *(_DWORD *)(a3 + 72);
  v9 = sub_36D7800(v8);
  v10 = *(const __m128i **)(a3 + 40);
  v11 = v9;
  v12 = v10[10].m128i_i64[0];
  v13 = *(_DWORD *)(v12 + 24);
  if ( v13 != 35 && v13 != 11 )
    sub_C64ED0("rowcol not constant", 1u);
  v14 = *(_QWORD *)(v12 + 96);
  v15 = *(_QWORD **)(v14 + 24);
  if ( *(_DWORD *)(v14 + 32) > 0x40u )
    v15 = (_QWORD *)*v15;
  v16 = _mm_loadu_si128(v10 + 5);
  v17 = *(_QWORD *)(a3 + 80);
  v35 = v17;
  v37 = v16;
  if ( v17 )
  {
    v32 = v11;
    sub_B96E90((__int64)&v35, v17, 1);
    v11 = v32;
  }
  v18 = *(_QWORD *)(a1 + 64);
  v36 = *(_DWORD *)(a3 + 72);
  v38 = sub_3400BD0(v18, v11, (__int64)&v35, 7, 0, 1u, v16, 0);
  v39 = v19;
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
  v20 = *(_QWORD *)(a3 + 80);
  v21 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a3 + 40) + 120LL));
  v35 = v20;
  v40 = v21;
  if ( v20 )
    sub_B96E90((__int64)&v35, v20, 1);
  v22 = *(_QWORD *)(a1 + 64);
  v36 = *(_DWORD *)(a3 + 72);
  v41 = sub_3400BD0(v22, (unsigned int)v15, (__int64)&v35, 7, 0, 1u, v16, 0);
  v42 = v24;
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
  v25 = *(_QWORD **)(a1 + 64);
  v26 = *(_QWORD *)(a3 + 48);
  v27 = *(unsigned int *)(a3 + 68);
  v43 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a3 + 40));
  v28 = sub_33E66D0(v25, a2, (__int64)&v33, v26, v27, v23, (unsigned __int64 *)&v37, 5);
  sub_34158F0(*(_QWORD *)(a1 + 64), a3, v28, v29, v30, v31);
  sub_3421DB0(v28);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a3);
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
}
