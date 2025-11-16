// Function: sub_36E5180
// Address: 0x36e5180
//
void __fastcall sub_36E5180(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // rsi
  int v9; // ecx
  __int64 v10; // rcx
  _QWORD *v11; // rsi
  __int64 v12; // rcx
  _QWORD *v13; // r14
  __int64 v14; // r8
  unsigned __int8 v15; // r13
  __m128i v16; // xmm0
  __m128i v17; // xmm1
  __int64 v18; // rdi
  __int64 v19; // r9
  int v20; // edx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r13
  __m128i v26; // xmm3
  __int64 v27; // rdi
  int v28; // edx
  __int64 v29; // rsi
  __int64 v30; // rdi
  int v31; // edx
  __int64 v32; // rsi
  __int64 v33; // rdi
  __int64 v34; // r9
  int v35; // edx
  __int64 v36; // [rsp-8h] [rbp-B8h]
  bool v37; // [rsp+Fh] [rbp-A1h]
  __int64 v38; // [rsp+10h] [rbp-A0h] BYREF
  int v39; // [rsp+18h] [rbp-98h]
  __int64 v40; // [rsp+20h] [rbp-90h] BYREF
  int v41; // [rsp+28h] [rbp-88h]
  __m128i v42; // [rsp+30h] [rbp-80h] BYREF
  __m128i v43; // [rsp+40h] [rbp-70h]
  unsigned __int8 *v44; // [rsp+50h] [rbp-60h]
  int v45; // [rsp+58h] [rbp-58h]
  unsigned __int8 *v46; // [rsp+60h] [rbp-50h]
  int v47; // [rsp+68h] [rbp-48h]
  unsigned __int8 *v48; // [rsp+70h] [rbp-40h]
  int v49; // [rsp+78h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v38 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v38, v5, 1);
  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_DWORD *)(a2 + 72);
  v8 = *(_QWORD *)(v6 + 40);
  v39 = v7;
  v9 = *(_DWORD *)(v8 + 24);
  if ( v9 != 11 && v9 != 35 )
    sub_C64ED0("First argument of 'llvm.nvvm.sub' must be a constant.", 1u);
  v10 = *(_QWORD *)(v8 + 96);
  v11 = *(_QWORD **)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = (_QWORD *)*v11;
  v12 = *(_QWORD *)(*(_QWORD *)v6 + 96LL);
  v13 = *(_QWORD **)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v13 = (_QWORD *)*v13;
  v14 = *(_QWORD *)(a2 + 80);
  v15 = (unsigned __int8)v11 & 7;
  if ( **(_WORD **)(a2 + 48) == 12 )
  {
    v42 = _mm_loadu_si128((const __m128i *)(v6 + 80));
    v26 = _mm_loadu_si128((const __m128i *)(v6 + 120));
    v37 = ((unsigned __int8)v11 & 8) != 0;
    v40 = v14;
    v43 = v26;
    if ( v14 )
    {
      sub_B96E90((__int64)&v40, v14, 1);
      v7 = *(_DWORD *)(a2 + 72);
    }
    v27 = *(_QWORD *)(a1 + 64);
    v41 = v7;
    v44 = sub_3400BD0(v27, v15, (__int64)&v40, 7, 0, 1u, a3, 0);
    v45 = v28;
    if ( v40 )
      sub_B91220((__int64)&v40, v40);
    v29 = *(_QWORD *)(a2 + 80);
    v40 = v29;
    if ( v29 )
      sub_B96E90((__int64)&v40, v29, 1);
    v30 = *(_QWORD *)(a1 + 64);
    v41 = *(_DWORD *)(a2 + 72);
    v46 = sub_3400BD0(v30, (_DWORD)v13 == 9578, (__int64)&v40, 7, 0, 1u, a3, 0);
    v47 = v31;
    if ( v40 )
      sub_B91220((__int64)&v40, v40);
    v32 = *(_QWORD *)(a2 + 80);
    v40 = v32;
    if ( v32 )
      sub_B96E90((__int64)&v40, v32, 1);
    v33 = *(_QWORD *)(a1 + 64);
    v41 = *(_DWORD *)(a2 + 72);
    v48 = sub_3400BD0(v33, v37, (__int64)&v40, 7, 0, 1u, a3, 0);
    v34 = v36;
    v49 = v35;
    if ( v40 )
      sub_B91220((__int64)&v40, v40);
    v21 = sub_33E66D0(
            *(_QWORD **)(a1 + 64),
            3725,
            (__int64)&v38,
            *(_QWORD *)(a2 + 48),
            *(unsigned int *)(a2 + 68),
            v34,
            (unsigned __int64 *)&v42,
            5);
  }
  else
  {
    v16 = _mm_loadu_si128((const __m128i *)(v6 + 80));
    v42 = v16;
    v17 = _mm_loadu_si128((const __m128i *)(v6 + 120));
    v40 = v14;
    v43 = v17;
    if ( v14 )
    {
      sub_B96E90((__int64)&v40, v14, 1);
      v7 = *(_DWORD *)(a2 + 72);
    }
    v18 = *(_QWORD *)(a1 + 64);
    v41 = v7;
    v44 = sub_3400BD0(v18, v15, (__int64)&v40, 7, 0, 1u, v16, 0);
    v45 = v20;
    if ( v40 )
      sub_B91220((__int64)&v40, v40);
    v21 = sub_33E66D0(
            *(_QWORD **)(a1 + 64),
            3727,
            (__int64)&v38,
            *(_QWORD *)(a2 + 48),
            *(unsigned int *)(a2 + 68),
            v19,
            (unsigned __int64 *)&v42,
            3);
  }
  v25 = v21;
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v21, v22, v23, v24);
  sub_3421DB0(v25);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
}
