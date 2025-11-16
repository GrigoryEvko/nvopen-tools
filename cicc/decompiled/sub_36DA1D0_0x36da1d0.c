// Function: sub_36DA1D0
// Address: 0x36da1d0
//
void __fastcall sub_36DA1D0(__int64 a1, __int64 a2, char a3, __m128i a4)
{
  __int64 v7; // rsi
  const __m128i *v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  _QWORD *v13; // rax
  bool v14; // zf
  __int64 v15; // rax
  __int64 v16; // rdi
  char v17; // r14
  _QWORD *v18; // rsi
  unsigned __int8 *v19; // r8
  const __m128i *v20; // rax
  __m128i v21; // xmm0
  __int32 v22; // edx
  int v23; // eax
  unsigned __int64 v24; // rcx
  __int64 v25; // r8
  _QWORD *v26; // r9
  unsigned __int64 *v27; // r10
  __int64 v28; // r11
  __int64 v29; // r13
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rcx
  _QWORD *v34; // rdx
  int v35; // eax
  unsigned __int64 v36; // rcx
  __int64 v37; // r8
  _QWORD *v38; // r9
  unsigned __int64 *v39; // r10
  __int64 v40; // r11
  __int64 v41; // r13
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // [rsp+0h] [rbp-70h] BYREF
  int v46; // [rsp+8h] [rbp-68h]
  __m128i v47; // [rsp+10h] [rbp-60h]
  __m128i v48; // [rsp+20h] [rbp-50h]
  __m128i v49; // [rsp+30h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 80);
  v45 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v45, v7, 1);
  v46 = *(_DWORD *)(a2 + 72);
  v8 = *(const __m128i **)(a2 + 40);
  v9 = *(_QWORD *)(v8[2].m128i_i64[1] + 96);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  v11 = v8[7].m128i_i64[1];
  if ( a3 )
  {
    v12 = *(_QWORD *)(v8[10].m128i_i64[0] + 96);
    v13 = *(_QWORD **)(v12 + 24);
    if ( *(_DWORD *)(v12 + 32) > 0x40u )
      v13 = (_QWORD *)*v13;
    v14 = v13 == 0;
    v15 = *(_QWORD *)(v11 + 96);
    v16 = *(_QWORD *)(a1 + 64);
    v17 = !v14;
    v18 = *(_QWORD **)(v15 + 24);
    if ( *(_DWORD *)(v15 + 32) > 0x40u )
      v18 = (_QWORD *)*v18;
    v19 = sub_3400BD0(v16, (__int64)v18, (__int64)&v45, 7, 0, 1u, a4, 0);
    v20 = *(const __m128i **)(a2 + 40);
    v21 = _mm_loadu_si128(v20 + 5);
    v48.m128i_i64[0] = (__int64)v19;
    v48.m128i_i32[2] = v22;
    v47 = v21;
    v49 = _mm_loadu_si128(v20);
    v23 = sub_36D67C0((int)v10, v17);
    v29 = sub_33E66D0(v26, v23, (__int64)&v45, v24, v25, (__int64)v26, v27, v28);
    sub_34158F0(*(_QWORD *)(a1 + 64), a2, v29, v30, v31, v32);
    sub_3421DB0(v29);
    sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  }
  else
  {
    v33 = *(_QWORD *)(v11 + 96);
    v34 = *(_QWORD **)(v33 + 24);
    if ( *(_DWORD *)(v33 + 32) > 0x40u )
      v34 = (_QWORD *)*v34;
    v47 = _mm_loadu_si128(v8 + 5);
    v48 = _mm_loadu_si128(v8);
    v35 = sub_36D67C0((int)v10, v34 != 0);
    v41 = sub_33E66D0(v38, v35, (__int64)&v45, v36, v37, (__int64)v38, v39, v40);
    sub_34158F0(*(_QWORD *)(a1 + 64), a2, v41, v42, v43, v44);
    sub_3421DB0(v41);
    sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  }
  if ( v45 )
    sub_B91220((__int64)&v45, v45);
}
