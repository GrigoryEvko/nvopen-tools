// Function: sub_3798B40
// Address: 0x3798b40
//
__int64 __fastcall sub_3798B40(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rsi
  __int64 v5; // rax
  _QWORD *v6; // r15
  __m128i v7; // xmm0
  unsigned __int16 *v8; // rax
  __int64 v9; // rdx
  int v10; // r14d
  __int64 v11; // rdx
  __int64 v12; // rsi
  unsigned int v13; // esi
  unsigned __int8 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  unsigned __int64 v17; // rcx
  __int64 v18; // r9
  __int64 v19; // rsi
  __int64 v20; // r8
  __int64 v21; // rcx
  unsigned __int8 *v22; // r14
  unsigned int v23; // edx
  unsigned __int64 v24; // r15
  bool v26; // al
  __int64 v27; // rcx
  __int128 v28; // [rsp-10h] [rbp-E0h]
  __int64 v29; // [rsp+0h] [rbp-D0h]
  unsigned __int8 *v30; // [rsp+10h] [rbp-C0h]
  __int64 v31; // [rsp+10h] [rbp-C0h]
  __int64 v32; // [rsp+10h] [rbp-C0h]
  __int64 v33; // [rsp+20h] [rbp-B0h]
  __int64 v34; // [rsp+40h] [rbp-90h] BYREF
  int v35; // [rsp+48h] [rbp-88h]
  __int16 v36; // [rsp+50h] [rbp-80h] BYREF
  __int64 v37; // [rsp+58h] [rbp-78h]
  unsigned __int16 v38; // [rsp+60h] [rbp-70h] BYREF
  __int64 v39; // [rsp+68h] [rbp-68h]
  __int16 v40; // [rsp+70h] [rbp-60h]
  __int64 v41; // [rsp+78h] [rbp-58h]
  __m128i v42; // [rsp+80h] [rbp-50h] BYREF
  __int64 v43; // [rsp+90h] [rbp-40h]
  __int64 v44; // [rsp+98h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(v3 + 40);
  v5 = sub_37946F0(a1, v4, *(_QWORD *)(v3 + 48));
  v6 = *(_QWORD **)(a1 + 8);
  v7 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v43 = v5;
  v8 = *(unsigned __int16 **)(a2 + 48);
  v44 = v9;
  v42 = v7;
  v10 = *v8;
  v11 = *((_QWORD *)v8 + 1);
  v36 = v10;
  v37 = v11;
  if ( (_WORD)v10 )
  {
    if ( (unsigned __int16)(v10 - 17) <= 0xD3u )
    {
      v11 = 0;
      LOWORD(v10) = word_4456580[v10 - 1];
    }
  }
  else
  {
    v32 = v11;
    v26 = sub_30070B0((__int64)&v36);
    v11 = v32;
    if ( v26 )
      LOWORD(v10) = sub_3009970((__int64)&v36, v4, v32, v27, (__int64)&v42);
  }
  v12 = *(_QWORD *)(a2 + 80);
  v39 = v11;
  v38 = v10;
  v40 = 1;
  v41 = 0;
  v34 = v12;
  if ( v12 )
    sub_B96E90((__int64)&v34, v12, 1);
  *((_QWORD *)&v28 + 1) = 2;
  *(_QWORD *)&v28 = &v42;
  v13 = *(_DWORD *)(a2 + 24);
  v35 = *(_DWORD *)(a2 + 72);
  v14 = sub_3411BE0(v6, v13, (__int64)&v34, &v38, 2, 2, v28);
  v16 = v15;
  v17 = (unsigned __int64)v14;
  if ( v34 )
  {
    v30 = v14;
    sub_B91220((__int64)&v34, v34);
    v17 = (unsigned __int64)v30;
  }
  sub_3760E70(a1, a2, 1, v17, 1);
  v18 = *(_QWORD *)(a1 + 8);
  v19 = *(_QWORD *)(a2 + 80);
  v20 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v21 = **(unsigned __int16 **)(a2 + 48);
  v42.m128i_i64[0] = v19;
  if ( v19 )
  {
    v29 = v21;
    v31 = v20;
    v33 = v18;
    sub_B96E90((__int64)&v42, v19, 1);
    v21 = v29;
    v20 = v31;
    v18 = v33;
  }
  v42.m128i_i32[2] = *(_DWORD *)(a2 + 72);
  v22 = sub_33FAF80(v18, 167, (__int64)&v42, v21, v20, v18, v7);
  v24 = v23 | v16 & 0xFFFFFFFF00000000LL;
  if ( v42.m128i_i64[0] )
    sub_B91220((__int64)&v42, v42.m128i_i64[0]);
  sub_3760E70(a1, a2, 0, (unsigned __int64)v22, v24);
  return 0;
}
