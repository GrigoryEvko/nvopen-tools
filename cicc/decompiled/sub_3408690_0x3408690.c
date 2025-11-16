// Function: sub_3408690
// Address: 0x3408690
//
void __fastcall sub_3408690(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int16 *a4,
        unsigned int a5,
        int a6,
        __m128i a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // rbx
  unsigned int v11; // r15d
  __int64 v12; // r8
  unsigned __int16 *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rsi
  __int128 v17; // rax
  __int64 v18; // r9
  __int64 v19; // rdx
  unsigned __int8 *v20; // r8
  __int64 v21; // rax
  __int64 v22; // r9
  unsigned __int8 **v23; // rax
  bool v24; // al
  int v25; // eax
  int v26; // eax
  __int64 v27; // rdx
  unsigned __int8 *v28; // [rsp+0h] [rbp-90h]
  __int16 v29; // [rsp+2h] [rbp-8Eh]
  __int64 v30; // [rsp+8h] [rbp-88h]
  int v31; // [rsp+10h] [rbp-80h]
  __int64 v32; // [rsp+10h] [rbp-80h]
  int v33; // [rsp+10h] [rbp-80h]
  __int64 v34; // [rsp+18h] [rbp-78h]
  int i; // [rsp+18h] [rbp-78h]
  __int64 v36; // [rsp+18h] [rbp-78h]
  __int64 v37; // [rsp+18h] [rbp-78h]
  __int64 v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+20h] [rbp-70h]
  __int16 v40; // [rsp+2Eh] [rbp-62h]
  __int128 v41; // [rsp+30h] [rbp-60h]
  unsigned __int16 v42; // [rsp+40h] [rbp-50h] BYREF
  __int64 v43; // [rsp+48h] [rbp-48h]
  __int64 v44; // [rsp+50h] [rbp-40h] BYREF
  int v45; // [rsp+58h] [rbp-38h]

  v10 = (__int64)a4;
  HIWORD(v11) = WORD1(a8);
  *(_QWORD *)&v41 = a2;
  *((_QWORD *)&v41 + 1) = a3;
  v40 = a8;
  v12 = a2;
  v39 = a9;
  v13 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v14 = *v13;
  v15 = *((_QWORD *)v13 + 1);
  v42 = v14;
  v43 = v15;
  if ( !a6 )
  {
    if ( (_WORD)v14 )
    {
      LOWORD(v14) = v14 - 176;
      if ( (unsigned __int16)v14 > 0x34u )
      {
LABEL_15:
        a4 = word_4456340;
        a6 = word_4456340[v42 - 1];
        goto LABEL_2;
      }
    }
    else
    {
      v32 = v12;
      v24 = sub_3007100((__int64)&v42);
      v12 = v32;
      if ( !v24 )
        goto LABEL_17;
    }
    v37 = v12;
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    v12 = v37;
    if ( v42 )
    {
      if ( (unsigned __int16)(v42 - 176) <= 0x34u )
      {
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
        v12 = v37;
      }
      goto LABEL_15;
    }
LABEL_17:
    v36 = v12;
    v25 = sub_3007130((__int64)&v42, v14);
    v12 = v36;
    a6 = v25;
  }
LABEL_2:
  if ( !(_WORD)a8 && !a9 )
  {
    if ( v42 )
    {
      v40 = word_4456580[v42 - 1];
    }
    else
    {
      v33 = a6;
      v38 = v12;
      v26 = sub_3009970((__int64)&v42, v14, a3, (__int64)a4, v12);
      a6 = v33;
      v12 = v38;
      v29 = HIWORD(v26);
      v40 = v26;
      v39 = v27;
    }
    HIWORD(v11) = v29;
  }
  v16 = *(_QWORD *)(v12 + 80);
  v44 = v16;
  if ( v16 )
  {
    v31 = a6;
    v34 = v12;
    sub_B96E90((__int64)&v44, v16, 1);
    a6 = v31;
    v12 = v34;
  }
  v45 = *(_DWORD *)(v12 + 72);
  for ( i = a6 + a5; i != a5; ++*(_DWORD *)(v10 + 8) )
  {
    *(_QWORD *)&v17 = sub_3400EE0((__int64)a1, a5, (__int64)&v44, 0, a7);
    LOWORD(v11) = v40;
    v20 = sub_3406EB0(a1, 0x9Eu, (__int64)&v44, v11, v39, v18, v41, v17);
    v21 = *(unsigned int *)(v10 + 8);
    v22 = v19;
    if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(v10 + 12) )
    {
      v28 = v20;
      v30 = v19;
      sub_C8D5F0(v10, (const void *)(v10 + 16), v21 + 1, 0x10u, (__int64)v20, v19);
      v21 = *(unsigned int *)(v10 + 8);
      v20 = v28;
      v22 = v30;
    }
    v23 = (unsigned __int8 **)(*(_QWORD *)v10 + 16 * v21);
    ++a5;
    *v23 = v20;
    v23[1] = (unsigned __int8 *)v22;
  }
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
}
