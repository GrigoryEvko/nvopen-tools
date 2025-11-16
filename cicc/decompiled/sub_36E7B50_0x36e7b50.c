// Function: sub_36E7B50
// Address: 0x36e7b50
//
void __fastcall sub_36E7B50(__int64 a1, unsigned int a2, unsigned int a3, int a4, __int64 a5)
{
  unsigned int v5; // eax
  __int64 v10; // rsi
  __int64 v11; // rdi
  unsigned int v12; // eax
  const __m128i *v13; // rdx
  unsigned int v14; // r8d
  __int64 v15; // rcx
  int v16; // eax
  __int64 v17; // rax
  __m128i v18; // xmm0
  __int64 v19; // rsi
  __int64 v20; // rdi
  int v21; // edx
  __int64 v22; // rsi
  __m128i v23; // xmm1
  __int64 v24; // rdi
  int v25; // edx
  __int64 v26; // rsi
  __int64 v27; // rdi
  int v28; // edx
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // r9
  int v32; // edx
  unsigned __int64 v33; // rcx
  _QWORD *v34; // rdi
  __int64 v35; // r8
  __int64 v36; // r12
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // [rsp+0h] [rbp-D0h]
  unsigned int v41; // [rsp+8h] [rbp-C8h]
  __int64 v43; // [rsp+10h] [rbp-C0h] BYREF
  int v44; // [rsp+18h] [rbp-B8h]
  __int64 v45; // [rsp+20h] [rbp-B0h] BYREF
  int v46; // [rsp+28h] [rbp-A8h]
  __m128i v47; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int8 *v48; // [rsp+40h] [rbp-90h]
  int v49; // [rsp+48h] [rbp-88h]
  __m128i v50; // [rsp+50h] [rbp-80h]
  unsigned __int8 *v51; // [rsp+60h] [rbp-70h]
  int v52; // [rsp+68h] [rbp-68h]
  unsigned __int8 *v53; // [rsp+70h] [rbp-60h]
  int v54; // [rsp+78h] [rbp-58h]
  unsigned __int8 *v55; // [rsp+80h] [rbp-50h]
  int v56; // [rsp+88h] [rbp-48h]
  __m128i v57; // [rsp+90h] [rbp-40h]

  v5 = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL);
  if ( v5 <= 0x47 || v5 == 72 && a3 > 1 )
    sub_C64ED0("immaldab is not supported on this architecture", 1u);
  v10 = *(_QWORD *)(a5 + 80);
  v43 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v43, v10, 1);
  v11 = *(_QWORD *)(a5 + 112);
  v44 = *(_DWORD *)(a5 + 72);
  v12 = sub_36D7800(v11);
  v13 = *(const __m128i **)(a5 + 40);
  v14 = v12;
  v15 = v13[10].m128i_i64[0];
  v16 = *(_DWORD *)(v15 + 24);
  if ( v16 != 35 && v16 != 11 )
    sub_C64ED0("rowcol not constant", 1u);
  v17 = *(_QWORD *)(v15 + 96);
  if ( *(_DWORD *)(v17 + 32) <= 0x40u )
    v40 = *(_QWORD *)(v17 + 24);
  else
    v40 = **(_QWORD **)(v17 + 24);
  v18 = _mm_loadu_si128(v13 + 5);
  v19 = *(_QWORD *)(a5 + 80);
  v45 = v19;
  v47 = v18;
  if ( v19 )
  {
    v41 = v14;
    sub_B96E90((__int64)&v45, v19, 1);
    v14 = v41;
  }
  v20 = *(_QWORD *)(a1 + 64);
  v46 = *(_DWORD *)(a5 + 72);
  v48 = sub_3400BD0(v20, v14, (__int64)&v45, 7, 0, 1u, v18, 0);
  v49 = v21;
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
  v22 = *(_QWORD *)(a5 + 80);
  v23 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a5 + 40) + 120LL));
  v45 = v22;
  v50 = v23;
  if ( v22 )
    sub_B96E90((__int64)&v45, v22, 1);
  v24 = *(_QWORD *)(a1 + 64);
  v46 = *(_DWORD *)(a5 + 72);
  v51 = sub_3400BD0(v24, a2, (__int64)&v45, 7, 0, 1u, v18, 0);
  v52 = v25;
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
  v26 = *(_QWORD *)(a5 + 80);
  v45 = v26;
  if ( v26 )
    sub_B96E90((__int64)&v45, v26, 1);
  v27 = *(_QWORD *)(a1 + 64);
  v46 = *(_DWORD *)(a5 + 72);
  v53 = sub_3400BD0(v27, (unsigned int)v40, (__int64)&v45, 7, 0, 1u, v18, 0);
  v54 = v28;
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
  v29 = *(_QWORD *)(a5 + 80);
  v45 = v29;
  if ( v29 )
    sub_B96E90((__int64)&v45, v29, 1);
  v30 = *(_QWORD *)(a1 + 64);
  v46 = *(_DWORD *)(a5 + 72);
  v55 = sub_3400BD0(v30, a3, (__int64)&v45, 7, 0, 1u, v18, 0);
  v56 = v32;
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
  v33 = *(_QWORD *)(a5 + 48);
  v34 = *(_QWORD **)(a1 + 64);
  v35 = *(unsigned int *)(a5 + 68);
  v57 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a5 + 40));
  v36 = sub_33E66D0(v34, a4, (__int64)&v43, v33, v35, v31, (unsigned __int64 *)&v47, 7);
  sub_34158F0(*(_QWORD *)(a1 + 64), a5, v36, v37, v38, v39);
  sub_3421DB0(v36);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a5);
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
}
