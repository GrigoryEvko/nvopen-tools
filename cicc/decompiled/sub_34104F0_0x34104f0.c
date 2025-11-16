// Function: sub_34104F0
// Address: 0x34104f0
//
__int64 __fastcall sub_34104F0(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4, __int64 a5, __int64 a6)
{
  __int128 *v7; // rbx
  __int64 v8; // rax
  unsigned __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  unsigned __int16 v13; // ax
  unsigned int v14; // r15d
  __int16 v15; // ax
  __int64 v16; // r8
  __int16 v17; // r14
  __int128 v18; // rax
  __int128 v19; // rax
  __int64 v20; // r9
  __int64 v21; // r8
  unsigned int v23; // eax
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // [rsp+0h] [rbp-80h]
  __int128 v29; // [rsp+0h] [rbp-80h]
  __int64 *v30; // [rsp+18h] [rbp-68h]
  __int64 v31; // [rsp+18h] [rbp-68h]
  unsigned int v32; // [rsp+20h] [rbp-60h]
  __int128 v33; // [rsp+20h] [rbp-60h]
  int v34; // [rsp+20h] [rbp-60h]
  unsigned __int16 v35; // [rsp+30h] [rbp-50h] BYREF
  __int64 v36; // [rsp+38h] [rbp-48h]
  __int64 v37; // [rsp+40h] [rbp-40h] BYREF
  int v38; // [rsp+48h] [rbp-38h]

  v7 = (__int128 *)a2;
  v8 = *(_QWORD *)(*(_QWORD *)a2 + 48LL) + 16LL * *(unsigned int *)(a2 + 8);
  v9 = *(_WORD *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v35 = v9;
  v36 = v10;
  if ( v9 )
  {
    if ( (unsigned __int16)(v9 - 176) > 0x34u )
      goto LABEL_3;
  }
  else if ( !sub_3007100((__int64)&v35) )
  {
LABEL_11:
    v23 = sub_3007130((__int64)&v35, a2);
    v24 = ((v23 | ((unsigned __int64)v23 >> 1)) >> 2)
        | v23
        | ((unsigned __int64)v23 >> 1)
        | ((((v23 | ((unsigned __int64)v23 >> 1)) >> 2) | v23 | ((unsigned __int64)v23 >> 1)) >> 4);
    v25 = (v24 >> 8) | v24;
    v11 = v25 >> 16;
    v12 = ((unsigned int)(v25 >> 16) | (unsigned int)v25) + 1;
    goto LABEL_12;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !v35 )
    goto LABEL_11;
  if ( (unsigned __int16)(v35 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_3:
  a2 = v35;
  v11 = v35 - 1;
  v12 = (((((((word_4456340[v11] | ((unsigned __int64)word_4456340[v11] >> 1)) >> 2)
           | word_4456340[v11]
           | ((unsigned __int64)word_4456340[v11] >> 1)) >> 4)
         | ((word_4456340[v11] | ((unsigned __int64)word_4456340[v11] >> 1)) >> 2)
         | word_4456340[v11]
         | ((unsigned __int64)word_4456340[v11] >> 1)) >> 8)
       | ((((word_4456340[v11] | ((unsigned __int64)word_4456340[v11] >> 1)) >> 2)
         | word_4456340[v11]
         | ((unsigned __int64)word_4456340[v11] >> 1)) >> 4)
       | ((word_4456340[v11] | ((unsigned __int64)word_4456340[v11] >> 1)) >> 2)
       | word_4456340[v11]
       | (unsigned int)((unsigned __int64)word_4456340[v11] >> 1))
      + 1;
  if ( v35 )
  {
    v28 = 0;
    v13 = word_4456580[v11];
    goto LABEL_5;
  }
LABEL_12:
  v34 = v12;
  v13 = sub_3009970((__int64)&v35, a2, v11, v12, a6);
  LODWORD(v12) = v34;
  v28 = v26;
LABEL_5:
  v32 = v12;
  v14 = v13;
  v30 = (__int64 *)a1[8];
  v15 = sub_2D43050(v13, v12);
  v16 = 0;
  if ( !v15 )
  {
    v15 = sub_3009400(v30, v14, v28, v32, 0);
    v16 = v27;
  }
  v31 = v16;
  v17 = v15;
  *(_QWORD *)&v18 = sub_3400EE0((__int64)a1, 0, a3, 0, a4);
  v33 = v18;
  v37 = 0;
  v38 = 0;
  *(_QWORD *)&v19 = sub_33F17F0(a1, 51, (__int64)&v37, v17, v31);
  v21 = v31;
  if ( v37 )
  {
    v29 = v19;
    sub_B91220((__int64)&v37, v37);
    v19 = v29;
    v21 = v31;
  }
  return sub_340F900(a1, 0xA0u, a3, v17, v21, v20, v19, *v7, v33);
}
