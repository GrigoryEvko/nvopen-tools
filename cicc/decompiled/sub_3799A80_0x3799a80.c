// Function: sub_3799A80
// Address: 0x3799a80
//
__int64 __fastcall sub_3799A80(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // r8
  _QWORD *v7; // r10
  __int64 v8; // rcx
  const __m128i *v9; // roff
  __m128i v10; // xmm0
  __int64 v11; // rdx
  unsigned __int16 *v12; // rdx
  int v13; // eax
  __int64 v14; // rdx
  unsigned __int16 v15; // ax
  __int64 v16; // rdx
  __int64 v17; // rsi
  unsigned __int8 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r15
  unsigned __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // rsi
  __int64 v24; // r8
  __int64 v25; // rcx
  unsigned __int8 *v26; // r14
  unsigned int v27; // edx
  unsigned __int64 v28; // r15
  __int128 v30; // [rsp-10h] [rbp-E0h]
  __int64 v31; // [rsp+8h] [rbp-C8h]
  _QWORD *v32; // [rsp+10h] [rbp-C0h]
  unsigned __int8 *v33; // [rsp+10h] [rbp-C0h]
  __int64 v34; // [rsp+10h] [rbp-C0h]
  __int64 v35; // [rsp+18h] [rbp-B8h]
  _QWORD *v36; // [rsp+18h] [rbp-B8h]
  __int64 v37; // [rsp+30h] [rbp-A0h] BYREF
  int v38; // [rsp+38h] [rbp-98h]
  __int16 v39; // [rsp+40h] [rbp-90h] BYREF
  __int64 v40; // [rsp+48h] [rbp-88h]
  unsigned __int16 v41; // [rsp+50h] [rbp-80h] BYREF
  __int64 v42; // [rsp+58h] [rbp-78h]
  __int16 v43; // [rsp+60h] [rbp-70h]
  __int64 v44; // [rsp+68h] [rbp-68h]
  __m128i v45; // [rsp+70h] [rbp-60h] BYREF
  __int64 v46; // [rsp+80h] [rbp-50h]
  __int64 v47; // [rsp+88h] [rbp-48h]
  __m128i v48; // [rsp+90h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(v3 + 40);
  v5 = sub_37946F0(a1, v4, *(_QWORD *)(v3 + 48));
  v7 = *(_QWORD **)(a1 + 8);
  v8 = v5;
  v9 = *(const __m128i **)(a2 + 40);
  v10 = _mm_loadu_si128(v9);
  v47 = v11;
  v46 = v5;
  v45 = v10;
  v12 = *(unsigned __int16 **)(a2 + 48);
  v48 = _mm_loadu_si128(v9 + 5);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  v39 = v13;
  v40 = v14;
  if ( (_WORD)v13 )
  {
    v15 = word_4456580[v13 - 1];
    v16 = 0;
  }
  else
  {
    v36 = v7;
    v15 = sub_3009970((__int64)&v39, v4, v14, v8, v6);
    v7 = v36;
  }
  v17 = *(_QWORD *)(a2 + 80);
  v41 = v15;
  v43 = 1;
  v42 = v16;
  v44 = 0;
  v37 = v17;
  if ( v17 )
  {
    v32 = v7;
    sub_B96E90((__int64)&v37, v17, 1);
    v7 = v32;
  }
  *((_QWORD *)&v30 + 1) = 3;
  *(_QWORD *)&v30 = &v45;
  v38 = *(_DWORD *)(a2 + 72);
  v18 = sub_3411BE0(v7, 0x91u, (__int64)&v37, &v41, 2, (__int64)&v37, v30);
  v20 = v19;
  v21 = (unsigned __int64)v18;
  if ( v37 )
  {
    v33 = v18;
    sub_B91220((__int64)&v37, v37);
    v21 = (unsigned __int64)v33;
  }
  sub_3760E70(a1, a2, 1, v21, 1);
  v22 = *(_QWORD *)(a1 + 8);
  v23 = *(_QWORD *)(a2 + 80);
  v24 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v25 = **(unsigned __int16 **)(a2 + 48);
  v45.m128i_i64[0] = v23;
  if ( v23 )
  {
    v31 = v25;
    v34 = v24;
    v35 = v22;
    sub_B96E90((__int64)&v45, v23, 1);
    v25 = v31;
    v24 = v34;
    v22 = v35;
  }
  v45.m128i_i32[2] = *(_DWORD *)(a2 + 72);
  v26 = sub_33FAF80(v22, 167, (__int64)&v45, v25, v24, v22, v10);
  v28 = v27 | v20 & 0xFFFFFFFF00000000LL;
  if ( v45.m128i_i64[0] )
    sub_B91220((__int64)&v45, v45.m128i_i64[0]);
  sub_3760E70(a1, a2, 0, (unsigned __int64)v26, v28);
  return 0;
}
