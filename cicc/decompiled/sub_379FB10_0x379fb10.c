// Function: sub_379FB10
// Address: 0x379fb10
//
unsigned __int8 *__fastcall sub_379FB10(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned __int64 *v5; // rax
  __int64 v6; // r11
  unsigned __int64 v7; // r13
  unsigned int v8; // ebx
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v10; // rax
  unsigned __int16 v11; // di
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned __int16 *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  _QWORD *v19; // r12
  unsigned int v20; // edx
  unsigned __int8 *result; // rax
  __int64 v22; // rax
  __int64 v23; // rsi
  _QWORD *v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // r9d
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // [rsp+8h] [rbp-88h]
  unsigned __int64 v31; // [rsp+10h] [rbp-80h]
  int v32; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v33; // [rsp+10h] [rbp-80h]
  unsigned int v34; // [rsp+20h] [rbp-70h] BYREF
  __int64 v35; // [rsp+28h] [rbp-68h]
  __int64 v36; // [rsp+30h] [rbp-60h] BYREF
  int v37; // [rsp+38h] [rbp-58h]
  __int64 v38; // [rsp+40h] [rbp-50h] BYREF
  __int64 v39; // [rsp+48h] [rbp-48h]
  __m128i v40; // [rsp+50h] [rbp-40h]

  v5 = *(unsigned __int64 **)(a2 + 40);
  v6 = *a1;
  v7 = *v5;
  v31 = *v5;
  v30 = v5[1];
  v8 = *((_DWORD *)v5 + 2);
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v13 = a1[1];
  if ( v9 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v38, v6, *(_QWORD *)(v13 + 64), v11, v12);
    LOWORD(v34) = v39;
    v35 = v40.m128i_i64[0];
  }
  else
  {
    v34 = v9(v6, *(_QWORD *)(v13 + 64), v11, v12);
    v35 = v29;
  }
  v14 = *a1;
  v15 = (unsigned __int16 *)(*(_QWORD *)(v7 + 48) + 16LL * v8);
  sub_2FE6CC0((__int64)&v38, *a1, *(_QWORD *)(a1[1] + 64), *v15, *((_QWORD *)v15 + 1));
  if ( (_BYTE)v38 != 7 )
  {
    v19 = (_QWORD *)a1[1];
    if ( (_WORD)v34 )
    {
      if ( (unsigned __int16)(v34 - 176) > 0x34u )
        goto LABEL_10;
    }
    else if ( !sub_3007100((__int64)&v34) )
    {
LABEL_6:
      v20 = sub_3007130((__int64)&v34, v14);
      return sub_3412A00(v19, a2, v20, v16, v17, v18, a3);
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !(_WORD)v34 )
      goto LABEL_6;
    if ( (unsigned __int16)(v34 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_10:
    v20 = word_4456340[(unsigned __int16)v34 - 1];
    return sub_3412A00(v19, a2, v20, v16, v17, v18, a3);
  }
  v22 = sub_379AB60((__int64)a1, v31, v30);
  v23 = *(_QWORD *)(a2 + 80);
  v24 = (_QWORD *)a1[1];
  v38 = v22;
  v25 = *(_QWORD *)(a2 + 40);
  v39 = v26;
  v27 = *(_DWORD *)(a2 + 28);
  v36 = v23;
  v40 = _mm_loadu_si128((const __m128i *)(v25 + 40));
  if ( v23 )
  {
    v32 = v27;
    sub_B96E90((__int64)&v36, v23, 1);
    v27 = v32;
  }
  v28 = *(unsigned int *)(a2 + 24);
  v37 = *(_DWORD *)(a2 + 72);
  result = sub_33FBA10(v24, v28, (__int64)&v36, v34, v35, v27, (__int64)&v38, 2);
  if ( v36 )
  {
    v33 = result;
    sub_B91220((__int64)&v36, v36);
    return v33;
  }
  return result;
}
