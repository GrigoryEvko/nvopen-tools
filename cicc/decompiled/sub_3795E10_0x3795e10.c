// Function: sub_3795E10
// Address: 0x3795e10
//
unsigned __int8 *__fastcall sub_3795E10(_QWORD *a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, int a7)
{
  __int64 v7; // r15
  unsigned __int64 *v8; // rax
  unsigned __int64 v9; // r12
  __int64 v10; // r13
  unsigned __int16 *v11; // rdx
  int v12; // eax
  __int64 v13; // r14
  unsigned __int16 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rsi
  __int64 v19; // rbx
  unsigned __int8 *v20; // r12
  bool v22; // al
  bool v23; // al
  __int64 v24; // rcx
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  __int64 v31; // [rsp+10h] [rbp-70h]
  __int16 v32; // [rsp+10h] [rbp-70h]
  __int64 v33; // [rsp+18h] [rbp-68h]
  __int64 v34; // [rsp+18h] [rbp-68h]
  __int64 v35; // [rsp+20h] [rbp-60h]
  unsigned __int64 v36; // [rsp+28h] [rbp-58h]
  __int16 v37; // [rsp+30h] [rbp-50h] BYREF
  __int64 v38; // [rsp+38h] [rbp-48h]
  __int64 v39; // [rsp+40h] [rbp-40h] BYREF
  __int64 v40; // [rsp+48h] [rbp-38h]

  v7 = a2;
  v8 = *(unsigned __int64 **)(a2 + 40);
  v9 = *v8;
  v10 = v8[1];
  v36 = *v8;
  v35 = *((unsigned int *)v8 + 2);
  v11 = (unsigned __int16 *)(*(_QWORD *)(*v8 + 48) + 16 * v35);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  v37 = v12;
  v38 = v13;
  if ( (_WORD)v12 )
  {
    a2 = (unsigned int)(v12 - 17);
    if ( (unsigned __int16)(v12 - 17) > 0xD3u )
      goto LABEL_3;
    LOWORD(v39) = v12;
    v40 = v13;
    if ( (unsigned __int16)(v12 - 176) <= 0x34u )
    {
      v28 = a5;
      v32 = v12;
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
      LOWORD(v12) = v32;
      a5 = v28;
    }
    v25 = word_4456340[(unsigned __int16)v12 - 1];
  }
  else
  {
    v30 = a5;
    v22 = sub_30070B0((__int64)&v37);
    a5 = v30;
    if ( !v22 )
      goto LABEL_3;
    v40 = v13;
    LOWORD(v39) = 0;
    v23 = sub_3007100((__int64)&v39);
    v24 = v30;
    if ( v23 )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      v24 = v30;
    }
    v31 = v24;
    v25 = sub_3007130((__int64)&v39, a2);
    a5 = v31;
  }
  if ( v25 == 1 )
  {
    v26 = *(unsigned __int16 *)(*(_QWORD *)(v36 + 48) + 16 * v35);
    if ( !(_WORD)v26 || !*(_QWORD *)(*a1 + 8 * v26 + 112) )
    {
      a2 = v9;
      v34 = a5;
      sub_37946F0((__int64)a1, v9, v10);
      a5 = v34;
    }
  }
LABEL_3:
  v14 = *(unsigned __int16 **)(v7 + 48);
  LODWORD(v15) = *v14;
  v16 = *((_QWORD *)v14 + 1);
  LOWORD(v39) = v15;
  v40 = v16;
  if ( (_WORD)v15 )
  {
    v17 = 0;
    LOWORD(v15) = word_4456580[(int)v15 - 1];
  }
  else
  {
    v15 = sub_3009970((__int64)&v39, a2, v16, a5, a6);
    a5 = v15;
    v17 = v27;
  }
  v18 = *(_QWORD *)(v7 + 80);
  v19 = a1[1];
  LOWORD(a5) = v15;
  v39 = v18;
  if ( v18 )
  {
    v29 = a5;
    v33 = v17;
    sub_B96E90((__int64)&v39, v18, 1);
    a5 = v29;
    v17 = v33;
  }
  LODWORD(v40) = *(_DWORD *)(v7 + 72);
  v20 = sub_33FAF80(v19, 234, (__int64)&v39, a5, v17, a7, a3);
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  return v20;
}
