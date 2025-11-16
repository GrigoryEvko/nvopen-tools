// Function: sub_36E72A0
// Address: 0x36e72a0
//
void __fastcall sub_36E72A0(__int64 a1, unsigned int a2, int a3, __int64 a4)
{
  __int64 v5; // rsi
  __int64 v9; // rdi
  unsigned int v10; // eax
  const __m128i *v11; // rdx
  unsigned int v12; // r8d
  __int64 v13; // rcx
  int v14; // eax
  __int64 v15; // rax
  __m128i v16; // xmm0
  __int64 v17; // rsi
  __int64 v18; // rdi
  int v19; // edx
  __int64 v20; // rsi
  __m128i v21; // xmm1
  __int64 v22; // rdi
  int v23; // edx
  __int64 v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // r9
  int v27; // edx
  _QWORD *v28; // rdi
  unsigned __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r13
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  unsigned int v35; // [rsp+4h] [rbp-BCh]
  __int64 v36; // [rsp+8h] [rbp-B8h]
  __int64 v37; // [rsp+10h] [rbp-B0h] BYREF
  int v38; // [rsp+18h] [rbp-A8h]
  __int64 v39; // [rsp+20h] [rbp-A0h] BYREF
  int v40; // [rsp+28h] [rbp-98h]
  __m128i v41; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int8 *v42; // [rsp+40h] [rbp-80h]
  int v43; // [rsp+48h] [rbp-78h]
  __m128i v44; // [rsp+50h] [rbp-70h]
  unsigned __int8 *v45; // [rsp+60h] [rbp-60h]
  int v46; // [rsp+68h] [rbp-58h]
  unsigned __int8 *v47; // [rsp+70h] [rbp-50h]
  int v48; // [rsp+78h] [rbp-48h]
  __m128i v49; // [rsp+80h] [rbp-40h]

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) <= 0x45u )
    sub_C64ED0("hmmaldab is not supported on this architecture", 1u);
  v5 = *(_QWORD *)(a4 + 80);
  v37 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v37, v5, 1);
  v9 = *(_QWORD *)(a4 + 112);
  v38 = *(_DWORD *)(a4 + 72);
  v10 = sub_36D7800(v9);
  v11 = *(const __m128i **)(a4 + 40);
  v12 = v10;
  v13 = v11[10].m128i_i64[0];
  v14 = *(_DWORD *)(v13 + 24);
  if ( v14 != 35 && v14 != 11 )
    sub_C64ED0("rowcol not constant", 1u);
  v15 = *(_QWORD *)(v13 + 96);
  if ( *(_DWORD *)(v15 + 32) <= 0x40u )
    v36 = *(_QWORD *)(v15 + 24);
  else
    v36 = **(_QWORD **)(v15 + 24);
  v16 = _mm_loadu_si128(v11 + 5);
  v17 = *(_QWORD *)(a4 + 80);
  v39 = v17;
  v41 = v16;
  if ( v17 )
  {
    v35 = v12;
    sub_B96E90((__int64)&v39, v17, 1);
    v12 = v35;
  }
  v18 = *(_QWORD *)(a1 + 64);
  v40 = *(_DWORD *)(a4 + 72);
  v42 = sub_3400BD0(v18, v12, (__int64)&v39, 7, 0, 1u, v16, 0);
  v43 = v19;
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  v20 = *(_QWORD *)(a4 + 80);
  v21 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a4 + 40) + 120LL));
  v39 = v20;
  v44 = v21;
  if ( v20 )
    sub_B96E90((__int64)&v39, v20, 1);
  v22 = *(_QWORD *)(a1 + 64);
  v40 = *(_DWORD *)(a4 + 72);
  v45 = sub_3400BD0(v22, a2, (__int64)&v39, 7, 0, 1u, v16, 0);
  v46 = v23;
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  v24 = *(_QWORD *)(a4 + 80);
  v39 = v24;
  if ( v24 )
    sub_B96E90((__int64)&v39, v24, 1);
  v25 = *(_QWORD *)(a1 + 64);
  v40 = *(_DWORD *)(a4 + 72);
  v47 = sub_3400BD0(v25, (unsigned int)v36, (__int64)&v39, 7, 0, 1u, v16, 0);
  v48 = v27;
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  v28 = *(_QWORD **)(a1 + 64);
  v29 = *(_QWORD *)(a4 + 48);
  v30 = *(unsigned int *)(a4 + 68);
  v49 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a4 + 40));
  v31 = sub_33E66D0(v28, a3, (__int64)&v37, v29, v30, v26, (unsigned __int64 *)&v41, 6);
  sub_34158F0(*(_QWORD *)(a1 + 64), a4, v31, v32, v33, v34);
  sub_3421DB0(v31);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a4);
  if ( v37 )
    sub_B91220((__int64)&v37, v37);
}
