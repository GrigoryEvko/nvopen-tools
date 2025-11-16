// Function: sub_378A160
// Address: 0x378a160
//
unsigned __int8 *__fastcall sub_378A160(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v7; // rsi
  unsigned __int16 *v8; // rdx
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r15
  unsigned int v12; // ebx
  unsigned int *v13; // rbx
  __int64 v14; // rax
  unsigned __int16 v15; // r13
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // r14
  _QWORD *v19; // r13
  __int128 v20; // rax
  __int64 v21; // r9
  __int64 v22; // r8
  unsigned __int8 *v23; // r10
  __int64 v24; // rax
  unsigned __int8 *v25; // rdx
  unsigned __int8 *v26; // r11
  unsigned __int64 v27; // rdx
  unsigned __int8 **v28; // rax
  unsigned __int8 *v29; // r14
  __int64 v31; // rdx
  __int128 v32; // [rsp-10h] [rbp-2C0h]
  unsigned int *i; // [rsp+18h] [rbp-298h]
  unsigned __int8 *v35; // [rsp+20h] [rbp-290h]
  unsigned __int8 *v36; // [rsp+28h] [rbp-288h]
  unsigned int v37; // [rsp+40h] [rbp-270h]
  __int16 v38; // [rsp+42h] [rbp-26Eh]
  __int64 v39; // [rsp+48h] [rbp-268h]
  __int64 v40; // [rsp+50h] [rbp-260h] BYREF
  int v41; // [rsp+58h] [rbp-258h]
  unsigned __int16 v42; // [rsp+60h] [rbp-250h] BYREF
  __int64 v43; // [rsp+68h] [rbp-248h]
  _BYTE *v44; // [rsp+70h] [rbp-240h] BYREF
  __int64 v45; // [rsp+78h] [rbp-238h]
  _BYTE v46[560]; // [rsp+80h] [rbp-230h] BYREF

  v7 = *(_QWORD *)(a2 + 80);
  v40 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v40, v7, 1);
  v8 = *(unsigned __int16 **)(a2 + 48);
  v41 = *(_DWORD *)(a2 + 72);
  v44 = v46;
  v45 = 0x2000000000LL;
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  v42 = v9;
  v43 = v10;
  if ( (_WORD)v9 )
  {
    v11 = 0;
    LOWORD(v9) = word_4456580[v9 - 1];
  }
  else
  {
    v9 = sub_3009970((__int64)&v42, v7, v10, a5, a6);
    v38 = HIWORD(v9);
    v11 = v31;
  }
  HIWORD(v12) = v38;
  LOWORD(v12) = v9;
  v37 = v12;
  v13 = *(unsigned int **)(a2 + 40);
  for ( i = &v13[10 * *(unsigned int *)(a2 + 64)]; i != v13; v13 += 10 )
  {
    v14 = *(_QWORD *)(*(_QWORD *)v13 + 48LL) + 16LL * v13[2];
    v15 = *(_WORD *)v14;
    v16 = *(_QWORD *)(v14 + 8);
    v42 = v15;
    v43 = v16;
    if ( v15 )
    {
      if ( (unsigned __int16)(v15 - 176) <= 0x34u )
      {
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      }
      v17 = word_4456340[v15 - 1];
    }
    else
    {
      if ( sub_3007100((__int64)&v42) )
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
      v17 = sub_3007130((__int64)&v42, v7);
    }
    v7 = v17;
    v39 = v17;
    v18 = 0;
    if ( v17 )
    {
      do
      {
        v19 = *(_QWORD **)(a1 + 8);
        *(_QWORD *)&v20 = sub_3400EE0((__int64)v19, v18, (__int64)&v40, 0, a3);
        v7 = 158;
        v23 = sub_3406EB0(v19, 0x9Eu, (__int64)&v40, v37, v11, v21, *(_OWORD *)v13, v20);
        v24 = (unsigned int)v45;
        v26 = v25;
        v27 = (unsigned int)v45 + 1LL;
        if ( v27 > HIDWORD(v45) )
        {
          v7 = (__int64)v46;
          v35 = v23;
          v36 = v26;
          sub_C8D5F0((__int64)&v44, v46, v27, 0x10u, v22, a7);
          v24 = (unsigned int)v45;
          v23 = v35;
          v26 = v36;
        }
        v28 = (unsigned __int8 **)&v44[16 * v24];
        ++v18;
        *v28 = v23;
        v28[1] = v26;
        LODWORD(v45) = v45 + 1;
      }
      while ( v18 != v39 );
    }
  }
  *((_QWORD *)&v32 + 1) = (unsigned int)v45;
  *(_QWORD *)&v32 = v44;
  v29 = sub_33FC220(
          *(_QWORD **)(a1 + 8),
          156,
          (__int64)&v40,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          a7,
          v32);
  if ( v44 != v46 )
    _libc_free((unsigned __int64)v44);
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
  return v29;
}
