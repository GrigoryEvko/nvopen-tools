// Function: sub_327AF80
// Address: 0x327af80
//
__int64 __fastcall sub_327AF80(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int16 *v8; // rax
  __int64 *v9; // rdi
  __int16 v10; // dx
  __int64 v11; // rax
  char v12; // r15
  __int64 result; // rax
  __int16 *v14; // rdx
  __int16 v15; // ax
  __int64 v16; // rdx
  unsigned int v17; // eax
  __m128i v18; // xmm0
  __int64 v19; // rax
  __int64 v20; // rcx
  int v21; // r9d
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // [rsp-10h] [rbp-C0h]
  __int64 v26; // [rsp-8h] [rbp-B8h]
  __int128 v27; // [rsp+0h] [rbp-B0h]
  int v28; // [rsp+1Ch] [rbp-94h]
  __int64 v29; // [rsp+20h] [rbp-90h]
  __int64 v30; // [rsp+28h] [rbp-88h]
  __int64 v31; // [rsp+28h] [rbp-88h]
  unsigned int v32; // [rsp+30h] [rbp-80h] BYREF
  __int64 v33; // [rsp+38h] [rbp-78h]
  _DWORD v34[4]; // [rsp+40h] [rbp-70h] BYREF
  char v35; // [rsp+50h] [rbp-60h]
  __int64 v36; // [rsp+60h] [rbp-50h] BYREF
  __int64 v37; // [rsp+68h] [rbp-48h]
  __int64 (__fastcall *v38)(unsigned __int64 *, const __m128i **, int); // [rsp+70h] [rbp-40h]
  __int64 (__fastcall *v39)(int **, unsigned int *); // [rsp+78h] [rbp-38h]

  v8 = *(__int16 **)(a1 + 48);
  v9 = *(__int64 **)(a2 + 40);
  v10 = *v8;
  v11 = *((_QWORD *)v8 + 1);
  LOWORD(v32) = v10;
  v33 = v11;
  v12 = *(_BYTE *)sub_2E79000(v9);
  if ( !(_WORD)v32 )
  {
    if ( sub_3007070((__int64)&v32) && !v12 )
    {
      if ( !sub_3007100((__int64)&v32) )
        goto LABEL_10;
      goto LABEL_28;
    }
    return 0;
  }
  if ( (unsigned __int16)(v32 - 2) > 7u && (unsigned __int16)(v32 - 17) > 0x6Cu && (unsigned __int16)(v32 - 176) > 0x1Fu
    || v12 )
  {
    return 0;
  }
  if ( (unsigned __int16)(v32 - 176) <= 0x34u )
  {
LABEL_28:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v32 )
    {
      if ( (unsigned __int16)(v32 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_13;
    }
LABEL_10:
    v28 = sub_3007130((__int64)&v32, a2);
    goto LABEL_14;
  }
LABEL_13:
  v28 = word_4456340[(unsigned __int16)v32 - 1];
LABEL_14:
  v14 = *(__int16 **)(a1 + 48);
  v15 = *v14;
  v16 = *((_QWORD *)v14 + 1);
  LOWORD(v36) = v15;
  v37 = v16;
  if ( v15 )
  {
    if ( (unsigned __int16)(v15 - 176) > 0x34u )
    {
LABEL_16:
      v17 = word_4456340[(unsigned __int16)v36 - 1];
      goto LABEL_17;
    }
  }
  else if ( !sub_3007100((__int64)&v36) )
  {
    goto LABEL_27;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v36 )
  {
    if ( (unsigned __int16)(v36 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_16;
  }
LABEL_27:
  v17 = sub_3007130((__int64)&v36, a2);
LABEL_17:
  v30 = v17;
  v29 = *(_QWORD *)(a1 + 96);
  v18 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  v38 = 0;
  v27 = (__int128)v18;
  v19 = sub_22077B0(0x18u);
  if ( v19 )
  {
    *(_DWORD *)v19 = v28;
    *(_QWORD *)(v19 + 8) = v29;
    *(_QWORD *)(v19 + 16) = v30;
  }
  v36 = v19;
  v39 = sub_325D530;
  v38 = sub_325ECC0;
  sub_327AB70((__int64)v34, 223, v32, v33, (__int64)&v36, a2, a3, a4);
  v21 = v26;
  if ( v38 )
    ((void (__fastcall *)(__int64 *, __int64 *, __int64, __int64, __int64, __int64, __int64, __int64))v38)(
      &v36,
      &v36,
      3,
      v20,
      v25,
      v26,
      v18.m128i_i64[0],
      v18.m128i_i64[1]);
  if ( !v35 )
    return 0;
  v22 = *(_QWORD *)(a1 + 80);
  v36 = v22;
  if ( v22 )
    sub_B96E90((__int64)&v36, v22, 1);
  LODWORD(v37) = *(_DWORD *)(a1 + 72);
  v23 = sub_33FAF80(a2, 223, (unsigned int)&v36, v34[0], v34[2], v21, v27);
  result = sub_33FB890(a2, v32, v33, v23, v24);
  if ( v36 )
  {
    v31 = result;
    sub_B91220((__int64)&v36, v36);
    return v31;
  }
  return result;
}
