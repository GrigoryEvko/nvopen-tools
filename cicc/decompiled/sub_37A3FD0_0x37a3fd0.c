// Function: sub_37A3FD0
// Address: 0x37a3fd0
//
_QWORD *__fastcall sub_37A3FD0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rax
  __int64 v5; // rsi
  __int16 v6; // dx
  __int64 v7; // rax
  __int64 v8; // r10
  __int64 v9; // rdx
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v11; // rsi
  unsigned int v12; // r15d
  __int16 v13; // ax
  unsigned int v14; // r11d
  __int64 v15; // r12
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  unsigned int v22; // r11d
  _QWORD *v23; // rsi
  __int64 v24; // rcx
  unsigned int v25; // r11d
  int v26; // edx
  _QWORD *v27; // r12
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // [rsp+20h] [rbp-D0h]
  __int64 v32; // [rsp+28h] [rbp-C8h]
  unsigned int v33; // [rsp+30h] [rbp-C0h]
  __int64 v34; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v35; // [rsp+48h] [rbp-A8h]
  __int64 v36; // [rsp+50h] [rbp-A0h] BYREF
  int v37; // [rsp+58h] [rbp-98h]
  unsigned int v38; // [rsp+60h] [rbp-90h] BYREF
  __int64 v39; // [rsp+68h] [rbp-88h]
  void *s; // [rsp+70h] [rbp-80h] BYREF
  __int64 v41; // [rsp+78h] [rbp-78h]
  _QWORD v42[14]; // [rsp+80h] [rbp-70h] BYREF

  v4 = *(__int16 **)(a2 + 48);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *v4;
  v7 = *((_QWORD *)v4 + 1);
  v36 = v5;
  v35 = v7;
  LOWORD(v34) = v6;
  if ( v5 )
    sub_B96E90((__int64)&v36, v5, 1);
  v8 = *a1;
  v9 = a1[1];
  v37 = *(_DWORD *)(a2 + 72);
  v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v8 + 592LL);
  if ( v10 == sub_2D56A50 )
  {
    v11 = v8;
    sub_2FE6CC0((__int64)&s, v8, *(_QWORD *)(v9 + 64), v34, v35);
    LOWORD(v38) = v41;
    v39 = v42[0];
  }
  else
  {
    v11 = *(_QWORD *)(v9 + 64);
    v38 = v10(v8, v11, v34, v35);
    v39 = v30;
  }
  if ( (_WORD)v34 )
  {
    if ( (unsigned __int16)(v34 - 176) > 0x34u )
      goto LABEL_7;
  }
  else if ( !sub_3007100((__int64)&v34) )
  {
    goto LABEL_24;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v34 )
  {
    if ( (unsigned __int16)(v34 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_7:
    v12 = word_4456340[(unsigned __int16)v34 - 1];
    v13 = v38;
    if ( !(_WORD)v38 )
      goto LABEL_8;
LABEL_25:
    if ( (unsigned __int16)(v13 - 176) > 0x34u )
      goto LABEL_26;
    goto LABEL_32;
  }
LABEL_24:
  v12 = sub_3007130((__int64)&v34, v11);
  v13 = v38;
  if ( (_WORD)v38 )
    goto LABEL_25;
LABEL_8:
  if ( !sub_3007100((__int64)&v38) )
  {
LABEL_9:
    v14 = sub_3007130((__int64)&v38, v11);
    goto LABEL_10;
  }
LABEL_32:
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)v38 )
    goto LABEL_9;
  if ( (unsigned __int16)(v38 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_26:
  v14 = word_4456340[(unsigned __int16)v38 - 1];
LABEL_10:
  v33 = v14;
  v15 = sub_379AB60((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v17 = v16;
  v18 = sub_379AB60((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v32 = v21;
  v31 = v18;
  s = v42;
  v22 = v33;
  v41 = 0x1000000000LL;
  if ( v33 > 0x10 )
  {
    sub_C8D5F0((__int64)&s, v42, v33, 4u, v19, v20);
    memset(s, 255, 4LL * v33);
    v22 = v33;
    v23 = s;
    LODWORD(v41) = v33;
  }
  else
  {
    if ( v33 )
    {
      v29 = 4LL * v33;
      if ( v29 )
      {
        if ( (unsigned int)v29 >= 8 )
        {
          *(_QWORD *)((char *)&v42[-1] + (unsigned int)v29) = -1;
          memset(v42, 0xFFu, 8LL * ((unsigned int)(v29 - 1) >> 3));
        }
        else if ( ((4 * (_BYTE)v33) & 4) != 0 )
        {
          LODWORD(v42[0]) = -1;
          *(_DWORD *)((char *)&v41 + (unsigned int)v29 + 4) = -1;
        }
        else if ( (_DWORD)v29 )
        {
          LOBYTE(v42[0]) = -1;
        }
      }
    }
    LODWORD(v41) = v33;
    v23 = v42;
  }
  if ( v12 )
  {
    v24 = 0;
    v25 = v22 - v12;
    do
    {
      v26 = *(_DWORD *)(*(_QWORD *)(a2 + 96) + v24);
      if ( v26 >= (int)v12 )
        v26 += v25;
      *(_DWORD *)((char *)v23 + v24) = v26;
      v24 += 4;
      v23 = s;
    }
    while ( v24 != 4LL * v12 );
  }
  v27 = sub_33FCE10(a1[1], v38, v39, (__int64)&v36, v15, v17, a3, v31, v32, v23, (unsigned int)v41);
  if ( s != v42 )
    _libc_free((unsigned __int64)s);
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
  return v27;
}
