// Function: sub_3799EA0
// Address: 0x3799ea0
//
__int64 __fastcall sub_3799EA0(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // r8
  const __m128i *v7; // rcx
  _QWORD *v8; // r10
  __m128i v9; // xmm0
  __int64 v10; // rdx
  unsigned __int16 *v11; // rdx
  int v12; // eax
  __int64 v13; // rdx
  unsigned __int16 v14; // ax
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned __int8 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r15
  unsigned __int64 v20; // rcx
  __int64 v21; // r9
  __int64 v22; // rsi
  __int64 v23; // r8
  __int64 v24; // rcx
  unsigned __int8 *v25; // r14
  unsigned int v26; // edx
  unsigned __int64 v27; // r15
  __int128 v29; // [rsp-10h] [rbp-D0h]
  __int64 v30; // [rsp+8h] [rbp-B8h]
  _QWORD *v31; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v32; // [rsp+10h] [rbp-B0h]
  __int64 v33; // [rsp+10h] [rbp-B0h]
  __int64 v34; // [rsp+18h] [rbp-A8h]
  _QWORD *v35; // [rsp+18h] [rbp-A8h]
  __int64 v36; // [rsp+30h] [rbp-90h] BYREF
  int v37; // [rsp+38h] [rbp-88h]
  __int16 v38; // [rsp+40h] [rbp-80h] BYREF
  __int64 v39; // [rsp+48h] [rbp-78h]
  unsigned __int16 v40; // [rsp+50h] [rbp-70h] BYREF
  __int64 v41; // [rsp+58h] [rbp-68h]
  __int16 v42; // [rsp+60h] [rbp-60h]
  __int64 v43; // [rsp+68h] [rbp-58h]
  __m128i v44; // [rsp+70h] [rbp-50h] BYREF
  __int64 v45; // [rsp+80h] [rbp-40h]
  __int64 v46; // [rsp+88h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(v3 + 40);
  v5 = sub_37946F0(a1, v4, *(_QWORD *)(v3 + 48));
  v7 = *(const __m128i **)(a2 + 40);
  v8 = *(_QWORD **)(a1 + 8);
  v9 = _mm_loadu_si128(v7);
  v46 = v10;
  v11 = *(unsigned __int16 **)(a2 + 48);
  v45 = v5;
  v44 = v9;
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  v38 = v12;
  v39 = v13;
  if ( (_WORD)v12 )
  {
    v14 = word_4456580[v12 - 1];
    v15 = 0;
  }
  else
  {
    v35 = v8;
    v14 = sub_3009970((__int64)&v38, v4, v13, (__int64)v7, v6);
    v8 = v35;
  }
  v16 = *(_QWORD *)(a2 + 80);
  v40 = v14;
  v42 = 1;
  v41 = v15;
  v43 = 0;
  v36 = v16;
  if ( v16 )
  {
    v31 = v8;
    sub_B96E90((__int64)&v36, v16, 1);
    v8 = v31;
  }
  *((_QWORD *)&v29 + 1) = 2;
  *(_QWORD *)&v29 = &v44;
  v37 = *(_DWORD *)(a2 + 72);
  v17 = sub_3411BE0(v8, 0x92u, (__int64)&v36, &v40, 2, (__int64)&v36, v29);
  v19 = v18;
  v20 = (unsigned __int64)v17;
  if ( v36 )
  {
    v32 = v17;
    sub_B91220((__int64)&v36, v36);
    v20 = (unsigned __int64)v32;
  }
  sub_3760E70(a1, a2, 1, v20, 1);
  v21 = *(_QWORD *)(a1 + 8);
  v22 = *(_QWORD *)(a2 + 80);
  v23 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v24 = **(unsigned __int16 **)(a2 + 48);
  v44.m128i_i64[0] = v22;
  if ( v22 )
  {
    v30 = v24;
    v33 = v23;
    v34 = v21;
    sub_B96E90((__int64)&v44, v22, 1);
    v24 = v30;
    v23 = v33;
    v21 = v34;
  }
  v44.m128i_i32[2] = *(_DWORD *)(a2 + 72);
  v25 = sub_33FAF80(v21, 167, (__int64)&v44, v24, v23, v21, v9);
  v27 = v26 | v19 & 0xFFFFFFFF00000000LL;
  if ( v44.m128i_i64[0] )
    sub_B91220((__int64)&v44, v44.m128i_i64[0]);
  sub_3760E70(a1, a2, 0, (unsigned __int64)v25, v27);
  return 0;
}
