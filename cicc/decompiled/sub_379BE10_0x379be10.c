// Function: sub_379BE10
// Address: 0x379be10
//
unsigned __int8 *__fastcall sub_379BE10(__int64 *a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned int v9; // edi
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  unsigned __int16 *v12; // rax
  int v13; // r14d
  __int64 v14; // r8
  __int16 *v15; // rdx
  unsigned __int16 v16; // si
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // r8
  unsigned __int16 v21; // dx
  __int64 v22; // rcx
  __int64 v23; // r9
  _QWORD *v24; // r15
  unsigned __int64 v25; // rdi
  unsigned int v26; // edx
  unsigned __int8 *v27; // r12
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned int v31; // edx
  __int64 v32; // r14
  unsigned int v33; // edx
  unsigned __int16 *v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rdx
  unsigned int v38; // eax
  __int64 v39; // rdx
  unsigned int v40; // [rsp+8h] [rbp-C8h]
  unsigned __int16 v41; // [rsp+Ch] [rbp-C4h]
  unsigned int v42; // [rsp+14h] [rbp-BCh]
  __int64 v43; // [rsp+18h] [rbp-B8h]
  __int128 v44; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v45; // [rsp+38h] [rbp-98h]
  __int64 v46; // [rsp+60h] [rbp-70h] BYREF
  int v47; // [rsp+68h] [rbp-68h]
  __int16 v48; // [rsp+70h] [rbp-60h] BYREF
  __int64 v49; // [rsp+78h] [rbp-58h]
  unsigned int v50; // [rsp+80h] [rbp-50h] BYREF
  __int64 v51; // [rsp+88h] [rbp-48h]
  __int64 v52; // [rsp+90h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD *)(a1[1] + 64);
  v46 = v4;
  v6 = v5;
  if ( v4 )
  {
    sub_B96E90((__int64)&v46, v4, 1);
    v6 = *(_QWORD *)(a1[1] + 64);
  }
  v7 = *a1;
  v47 = *(_DWORD *)(a2 + 72);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_DWORD *)(v8 + 8);
  v10 = _mm_loadu_si128((const __m128i *)v8);
  v11 = _mm_loadu_si128((const __m128i *)(v8 + 40));
  v43 = *(_QWORD *)v8;
  v12 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v8 + 48LL) + 16LL * v9);
  v42 = v9;
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  v45 = v10.m128i_u64[1];
  v44 = (__int128)v11;
  v48 = v13;
  v49 = v14;
  sub_2FE6CC0((__int64)&v50, v7, v6, (unsigned __int16)v13, v14);
  if ( (_BYTE)v50 == 7 )
  {
    v43 = sub_379AB60((__int64)a1, v10.m128i_u64[0], v10.m128i_i64[1]);
    v32 = v31;
    v42 = v31;
    v45 = v31 | v10.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v44 = sub_379AB60((__int64)a1, v11.m128i_u64[0], v11.m128i_i64[1]);
    *((_QWORD *)&v44 + 1) = v33 | v11.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v34 = (unsigned __int16 *)(*(_QWORD *)(v43 + 48) + 16 * v32);
    v13 = *v34;
    v35 = *((_QWORD *)v34 + 1);
    v48 = *v34;
    v49 = v35;
  }
  v15 = *(__int16 **)(a2 + 48);
  v16 = *v15;
  v17 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v17 == sub_2D56A50 )
  {
    v18 = v16;
    v19 = *a1;
    sub_2FE6CC0((__int64)&v50, *a1, v5, v18, *((_QWORD *)v15 + 1));
    v21 = v51;
    LOWORD(v50) = v51;
    v51 = v52;
  }
  else
  {
    v36 = *((_QWORD *)v15 + 1);
    v37 = v16;
    v19 = v5;
    v38 = v17(*a1, v5, v37, v36);
    v51 = v39;
    v21 = v38;
    v50 = v38;
  }
  if ( v21 )
  {
    v22 = word_4456340[v21 - 1];
    LOBYTE(v19) = (unsigned __int16)(v21 - 176) <= 0x34u;
    if ( (_WORD)v13 )
      goto LABEL_9;
  }
  else
  {
    v29 = sub_3007240((__int64)&v50);
    v21 = 0;
    v22 = v29;
    v19 = HIDWORD(v29);
    if ( (_WORD)v13 )
    {
LABEL_9:
      v23 = (unsigned int)(v13 - 176);
      v24 = (_QWORD *)a1[1];
      LOBYTE(v25) = (unsigned __int16)(v13 - 176) <= 0x34u;
      if ( word_4456340[(unsigned __int16)v13 - 1] != (_DWORD)v22 )
        goto LABEL_10;
      goto LABEL_21;
    }
  }
  v40 = v22;
  v41 = v21;
  v30 = sub_3007240((__int64)&v48);
  v22 = v40;
  v21 = v41;
  v19 = (unsigned __int8)v19;
  v24 = (_QWORD *)a1[1];
  v25 = HIDWORD(v30);
  if ( (_DWORD)v30 != v40 )
    goto LABEL_10;
LABEL_21:
  if ( (_BYTE)v19 != (_BYTE)v25 )
  {
LABEL_10:
    if ( v21 )
    {
      if ( (unsigned __int16)(v21 - 176) > 0x34u )
        goto LABEL_18;
    }
    else if ( !sub_3007100((__int64)&v50) )
    {
LABEL_12:
      v26 = sub_3007130((__int64)&v50, v19);
LABEL_13:
      v27 = sub_3412A00(v24, a2, v26, v22, v20, v23, v10);
      goto LABEL_14;
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !(_WORD)v50 )
      goto LABEL_12;
    if ( (unsigned __int16)(v50 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_18:
    v26 = word_4456340[(unsigned __int16)v50 - 1];
    goto LABEL_13;
  }
  v27 = sub_3406EB0(
          v24,
          *(_DWORD *)(a2 + 24),
          (__int64)&v46,
          v50,
          v51,
          v23,
          __PAIR128__(v42 | v45 & 0xFFFFFFFF00000000LL, v43),
          v44);
LABEL_14:
  if ( v46 )
    sub_B91220((__int64)&v46, v46);
  return v27;
}
