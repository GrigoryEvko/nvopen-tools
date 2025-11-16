// Function: sub_37A7A50
// Address: 0x37a7a50
//
__int64 __fastcall sub_37A7A50(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int16 *v5; // rdx
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // r15
  unsigned int v9; // esi
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned __int16 v12; // bx
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v18; // r8
  __int64 v19; // r9
  __int16 v20; // ax
  __int64 v21; // rdx
  __int64 v22; // rsi
  int v23; // edx
  __int64 v24; // rdi
  __int64 v25; // rax
  unsigned int v26; // r13d
  _QWORD *v27; // rax
  __int64 v28; // rsi
  _QWORD *v29; // rcx
  _QWORD *i; // rdx
  __m128i v31; // xmm0
  unsigned int v32; // edx
  __int64 v33; // r13
  __int64 v34; // rsi
  __int64 v35; // rbx
  _QWORD *v36; // r12
  __int128 v37; // rax
  __int64 v38; // r9
  unsigned __int8 *v39; // rax
  unsigned __int8 **v40; // rbx
  int v41; // edx
  __int64 v42; // r12
  __int64 v44; // rdx
  __int128 v45; // [rsp-10h] [rbp-1F0h]
  __int64 v46; // [rsp+8h] [rbp-1D8h]
  __int64 v48; // [rsp+18h] [rbp-1C8h]
  unsigned int v49; // [rsp+20h] [rbp-1C0h]
  __int16 v50; // [rsp+22h] [rbp-1BEh]
  int v51; // [rsp+30h] [rbp-1B0h]
  int v52; // [rsp+34h] [rbp-1ACh]
  __int128 v54; // [rsp+40h] [rbp-1A0h]
  __int64 v55; // [rsp+70h] [rbp-170h] BYREF
  __int64 v56; // [rsp+78h] [rbp-168h]
  __int64 v57; // [rsp+80h] [rbp-160h] BYREF
  __int64 v58; // [rsp+88h] [rbp-158h]
  __int64 v59; // [rsp+90h] [rbp-150h] BYREF
  int v60; // [rsp+98h] [rbp-148h]
  _QWORD *v61; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v62; // [rsp+A8h] [rbp-138h]
  _QWORD v63[38]; // [rsp+B0h] [rbp-130h] BYREF

  v5 = *(unsigned __int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  LOWORD(v55) = v6;
  v56 = v7;
  if ( (_WORD)v6 )
  {
    v8 = 0;
    LOWORD(v6) = word_4456580[v6 - 1];
  }
  else
  {
    v6 = sub_3009970((__int64)&v55, a2, v7, a4, a5);
    v50 = HIWORD(v6);
    v8 = v44;
  }
  HIWORD(v9) = v50;
  LOWORD(v9) = v6;
  v49 = v9;
  v10 = *(_QWORD *)(a2 + 80);
  v11 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v59 = v10;
  LOWORD(v57) = v12;
  v58 = v13;
  if ( v10 )
    sub_B96E90((__int64)&v59, v10, 1);
  v14 = *a1;
  v15 = a1[1];
  v16 = *(unsigned int *)(a2 + 64);
  v60 = *(_DWORD *)(a2 + 72);
  v17 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v14 + 592LL);
  if ( v17 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v61, v14, *(_QWORD *)(v15 + 64), v57, v58);
    v20 = v62;
    v21 = v63[0];
  }
  else
  {
    v20 = v17(v14, *(_QWORD *)(v15 + 64), v57, v58);
  }
  v22 = (unsigned __int16)v55;
  if ( v20 != (_WORD)v55 )
    goto LABEL_13;
  if ( !(_WORD)v55 && v21 != v56 )
    goto LABEL_40;
  if ( (unsigned int)v16 > 1 )
  {
    v23 = 1;
    v24 = *(_QWORD *)(a2 + 40);
    v25 = v24 + 40;
    while ( *(_DWORD *)(*(_QWORD *)v25 + 24LL) == 51 )
    {
      ++v23;
      v25 += 40;
      if ( (_DWORD)v16 == v23 )
        goto LABEL_52;
    }
LABEL_13:
    if ( (_WORD)v55 )
    {
      LOWORD(v22) = v55 - 176;
      if ( (unsigned __int16)(v55 - 176) > 0x34u )
        goto LABEL_15;
      goto LABEL_47;
    }
LABEL_40:
    if ( !sub_3007100((__int64)&v55) )
    {
LABEL_41:
      v26 = sub_3007130((__int64)&v55, v22);
LABEL_16:
      v27 = v63;
      v28 = 0x1000000000LL;
      v29 = v63;
      v61 = v63;
      v62 = 0x1000000000LL;
      if ( v26 )
      {
        if ( v26 > 0x10uLL )
        {
          v28 = (__int64)v63;
          sub_C8D5F0((__int64)&v61, v63, v26, 0x10u, v18, v19);
          v29 = v61;
          v27 = &v61[2 * (unsigned int)v62];
        }
        for ( i = &v29[2 * v26]; i != v27; v27 += 2 )
        {
          if ( v27 )
          {
            *v27 = 0;
            *((_DWORD *)v27 + 2) = 0;
          }
        }
        LODWORD(v62) = v26;
      }
      if ( v12 )
      {
        if ( (unsigned __int16)(v12 - 176) <= 0x34u )
        {
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
          sub_CA17B0(
            "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " MVT::getVectorElementCount() instead");
        }
        v51 = word_4456340[v12 - 1];
        if ( !(_DWORD)v16 )
          goto LABEL_34;
      }
      else
      {
        if ( sub_3007100((__int64)&v57) )
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
        v51 = sub_3007130((__int64)&v57, v28);
        if ( !(_DWORD)v16 )
          goto LABEL_34;
      }
      v48 = 0;
      v52 = 0;
      v46 = 40 * v16;
      do
      {
        v31 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v48));
        *(_QWORD *)&v54 = sub_379AB60((__int64)a1, v31.m128i_u64[0], v31.m128i_i64[1]);
        *((_QWORD *)&v54 + 1) = v32 | v31.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        if ( v51 )
        {
          v33 = 0;
          do
          {
            v34 = v33;
            v35 = (unsigned int)(v52 + v33++);
            v36 = (_QWORD *)a1[1];
            *(_QWORD *)&v37 = sub_3400EE0((__int64)v36, v34, (__int64)&v59, 0, v31);
            v39 = sub_3406EB0(v36, 0x9Eu, (__int64)&v59, v49, v8, v38, v54, v37);
            v40 = (unsigned __int8 **)&v61[2 * v35];
            *v40 = v39;
            *((_DWORD *)v40 + 2) = v41;
          }
          while ( v33 != v51 );
          v52 += v51;
        }
        v48 += 40;
      }
      while ( v46 != v48 );
LABEL_34:
      *((_QWORD *)&v45 + 1) = (unsigned int)v62;
      *(_QWORD *)&v45 = v61;
      v42 = (__int64)sub_33FC220((_QWORD *)a1[1], 156, (__int64)&v59, v55, v56, v19, v45);
      if ( v61 != v63 )
        _libc_free((unsigned __int64)v61);
      goto LABEL_36;
    }
LABEL_47:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !(_WORD)v55 )
      goto LABEL_41;
    if ( (unsigned __int16)(v55 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_15:
    v26 = word_4456340[(unsigned __int16)v55 - 1];
    goto LABEL_16;
  }
  if ( (_DWORD)v16 != 1 )
    goto LABEL_13;
  v24 = *(_QWORD *)(a2 + 40);
LABEL_52:
  v42 = sub_379AB60((__int64)a1, *(_QWORD *)v24, *(_QWORD *)(v24 + 8));
LABEL_36:
  if ( v59 )
    sub_B91220((__int64)&v59, v59);
  return v42;
}
