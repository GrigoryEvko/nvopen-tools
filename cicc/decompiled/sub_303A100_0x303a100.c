// Function: sub_303A100
// Address: 0x303a100
//
__int64 __fastcall sub_303A100(__int64 a1, __int64 a2, unsigned int a3, int a4)
{
  __int64 *v6; // rcx
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rsi
  __int64 result; // rax
  unsigned __int16 v11; // bx
  __int64 v12; // rax
  __int64 v13; // r10
  __int64 v14; // rcx
  int v15; // edx
  __int64 v16; // r10
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int128 v19; // rax
  __int128 v20; // rax
  int v21; // r9d
  __int128 v22; // [rsp-40h] [rbp-C0h]
  int v23; // [rsp+18h] [rbp-68h]
  __int128 v24; // [rsp+20h] [rbp-60h]
  int v25; // [rsp+30h] [rbp-50h]
  __int128 v26; // [rsp+30h] [rbp-50h]
  __int64 v27; // [rsp+30h] [rbp-50h]
  __int64 v28; // [rsp+40h] [rbp-40h] BYREF
  __int64 v29; // [rsp+48h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 40);
  v7 = *v6;
  v8 = v6[1];
  if ( *(_WORD *)(*(_QWORD *)(*v6 + 48) + 16LL * *((unsigned int *)v6 + 2)) != 37 )
    return a2;
  v9 = *(_QWORD *)(a2 + 48);
  if ( *(_WORD *)(v9 + 16LL * a3) != 37 )
    return a2;
  v11 = *(_WORD *)v9;
  v12 = *(_QWORD *)(v9 + 8);
  v24 = (__int128)_mm_loadu_si128((const __m128i *)(v6 + 5));
  LOWORD(v28) = v11;
  v29 = v12;
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v13 = word_4456340[v11 - 1];
  }
  else
  {
    if ( sub_3007100((__int64)&v28) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    v13 = (unsigned int)sub_3007130((__int64)&v28, v9);
  }
  v14 = 0;
  v15 = 0;
  v16 = 4 * v13;
  v17 = *(_QWORD *)(a2 + 96);
  if ( v16 )
  {
    do
    {
      if ( *(_DWORD *)(v17 + v14) != -1 )
        v15 |= *(_DWORD *)(v17 + v14) << v14;
      v14 += 4;
    }
    while ( v16 != v14 );
    LODWORD(v16) = v15;
  }
  v18 = *(_QWORD *)(a2 + 80);
  v28 = v18;
  if ( v18 )
  {
    v25 = v16;
    sub_B96E90((__int64)&v28, v18, 1);
    LODWORD(v16) = v25;
  }
  v23 = v16;
  LODWORD(v29) = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v19 = sub_3400BD0(a4, 0, (unsigned int)&v28, 7, 0, 0, 0);
  v26 = v19;
  *(_QWORD *)&v20 = sub_3400BD0(a4, v23, (unsigned int)&v28, 7, 0, 0, 0);
  *((_QWORD *)&v22 + 1) = v8;
  *(_QWORD *)&v22 = v7;
  result = sub_33FC130(a4, 537, (unsigned int)&v28, 37, 0, v21, v22, v24, v20, v26);
  if ( v28 )
  {
    v27 = result;
    sub_B91220((__int64)&v28, v28);
    return v27;
  }
  return result;
}
