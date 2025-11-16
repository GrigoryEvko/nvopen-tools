// Function: sub_37A28B0
// Address: 0x37a28b0
//
__m128i *__fastcall sub_37A28B0(__int64 *a1, unsigned __int64 a2)
{
  unsigned int v2; // ebx
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r11
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rdx
  __m128i v10; // xmm0
  __int64 v11; // rax
  unsigned __int16 v12; // cx
  __int128 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rsi
  unsigned __int16 *v16; // rdx
  __int64 v17; // rcx
  int v18; // eax
  unsigned __int16 v19; // ax
  __int64 v20; // rdx
  unsigned int v21; // r14d
  int v22; // eax
  __int64 v23; // r8
  unsigned __int8 *v24; // rax
  __int64 v25; // r11
  unsigned int v26; // edx
  __m128i *v27; // r14
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 *v31; // [rsp+8h] [rbp-B8h]
  __int128 v32; // [rsp+10h] [rbp-B0h]
  __int64 v33; // [rsp+20h] [rbp-A0h]
  int v34; // [rsp+28h] [rbp-98h]
  unsigned int v35; // [rsp+28h] [rbp-98h]
  char v36; // [rsp+2Ch] [rbp-94h]
  __int128 v37; // [rsp+30h] [rbp-90h]
  unsigned int v38; // [rsp+50h] [rbp-70h] BYREF
  __int64 v39; // [rsp+58h] [rbp-68h]
  unsigned __int16 v40; // [rsp+60h] [rbp-60h] BYREF
  __int64 v41; // [rsp+68h] [rbp-58h]
  __int64 v42; // [rsp+70h] [rbp-50h] BYREF
  int v43; // [rsp+78h] [rbp-48h]
  __int64 v44; // [rsp+80h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v42, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    LOWORD(v38) = v43;
    v39 = v44;
  }
  else
  {
    v38 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v39 = v30;
  }
  v9 = *(_QWORD *)(a2 + 40);
  v10 = _mm_loadu_si128((const __m128i *)(v9 + 120));
  v11 = *(_QWORD *)(*(_QWORD *)(v9 + 120) + 48LL) + 16LL * *(unsigned int *)(v9 + 128);
  v12 = *(_WORD *)v11;
  v41 = *(_QWORD *)(v11 + 8);
  v40 = v12;
  *(_QWORD *)&v13 = sub_379AB60((__int64)a1, *(_QWORD *)(v9 + 160), *(_QWORD *)(v9 + 168));
  v15 = *(_QWORD *)(a2 + 80);
  v32 = v13;
  LOBYTE(v13) = *(_BYTE *)(a2 + 33);
  v42 = v15;
  v36 = ((unsigned __int8)v13 >> 2) & 3;
  if ( v15 )
    sub_B96E90((__int64)&v42, v15, 1);
  v43 = *(_DWORD *)(a2 + 72);
  if ( (_WORD)v38 )
  {
    if ( (unsigned __int16)(v38 - 176) > 0x34u )
      goto LABEL_7;
  }
  else if ( !sub_3007100((__int64)&v38) )
  {
    goto LABEL_10;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)v38 )
  {
LABEL_10:
    v17 = (unsigned int)sub_3007130((__int64)&v38, v15);
    v18 = v40;
    if ( !v40 )
      goto LABEL_8;
    goto LABEL_11;
  }
  if ( (unsigned __int16)(v38 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_7:
  v16 = word_4456340;
  v17 = word_4456340[(unsigned __int16)v38 - 1];
  v18 = v40;
  if ( !v40 )
  {
LABEL_8:
    v34 = v17;
    v19 = sub_3009970((__int64)&v40, v15, (__int64)v16, v17, v14);
    LODWORD(v17) = v34;
    v33 = v20;
    goto LABEL_12;
  }
LABEL_11:
  v33 = 0;
  v19 = word_4456580[v18 - 1];
LABEL_12:
  v35 = v17;
  v21 = v19;
  v31 = *(__int64 **)(a1[1] + 64);
  LOWORD(v22) = sub_2D43050(v19, v17);
  v23 = 0;
  if ( !(_WORD)v22 )
  {
    v22 = sub_3009400(v31, v21, v33, v35, 0);
    HIWORD(v2) = HIWORD(v22);
    v23 = v29;
  }
  LOWORD(v2) = v22;
  v24 = sub_3790540((__int64)a1, v10.m128i_i64[0], v10.m128i_i64[1], v2, v23, 1, v10);
  v25 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)&v37 = v24;
  *((_QWORD *)&v37 + 1) = v26 | v10.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v27 = sub_33E8F60(
          (__int64 *)a1[1],
          v38,
          v39,
          (__int64)&v42,
          *(_QWORD *)v25,
          *(_QWORD *)(v25 + 8),
          *(_QWORD *)(v25 + 40),
          *(_QWORD *)(v25 + 48),
          *(_OWORD *)(v25 + 80),
          v37,
          v32,
          *(unsigned __int16 *)(a2 + 96),
          *(_QWORD *)(a2 + 104),
          *(const __m128i **)(a2 + 112),
          (*(_WORD *)(a2 + 32) >> 7) & 7,
          v36,
          (*(_BYTE *)(a2 + 33) & 0x10) != 0);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v27, 1);
  if ( v42 )
    sub_B91220((__int64)&v42, v42);
  return v27;
}
