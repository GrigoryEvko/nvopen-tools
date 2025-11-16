// Function: sub_3487190
// Address: 0x3487190
//
__int64 __fastcall sub_3487190(
        unsigned int *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        _QWORD **a6,
        __m128i a7,
        unsigned int a8,
        unsigned __int8 a9)
{
  __int64 v9; // r10
  __int64 v12; // rax
  unsigned __int16 v13; // dx
  __int64 v14; // rax
  __int64 result; // rax
  bool v16; // al
  bool v17; // al
  unsigned int v18; // eax
  unsigned __int64 v19; // rdx
  __int64 v21; // [rsp+8h] [rbp-68h]
  __int64 v22; // [rsp+10h] [rbp-60h]
  unsigned __int8 v24; // [rsp+18h] [rbp-58h]
  unsigned __int16 v25; // [rsp+20h] [rbp-50h] BYREF
  __int64 v26; // [rsp+28h] [rbp-48h]
  unsigned __int64 v27; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-38h]

  v9 = a4;
  v12 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  v25 = v13;
  v26 = v14;
  if ( v13 )
  {
    if ( (unsigned __int16)(v13 - 17) > 0x9Eu )
    {
LABEL_3:
      v28 = 1;
      v27 = 1;
      goto LABEL_4;
    }
LABEL_17:
    v18 = word_4456340[v25 - 1];
    v28 = v18;
    if ( v18 > 0x40 )
      goto LABEL_18;
    goto LABEL_11;
  }
  v16 = sub_30070D0((__int64)&v25);
  v9 = a4;
  if ( !v16 )
    goto LABEL_3;
  v17 = sub_3007100((__int64)&v25);
  v9 = a4;
  if ( v17 )
  {
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    v9 = a4;
    if ( v25 )
    {
      if ( (unsigned __int16)(v25 - 176) <= 0x34u )
      {
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
        v9 = a4;
      }
      goto LABEL_17;
    }
  }
  v22 = v9;
  v18 = sub_3007130((__int64)&v25, a2);
  v9 = v22;
  v28 = v18;
  if ( v18 > 0x40 )
  {
LABEL_18:
    v21 = v9;
    sub_C43690((__int64)&v27, -1, 1);
    v9 = v21;
    goto LABEL_4;
  }
LABEL_11:
  v19 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v18;
  if ( !v18 )
    v19 = 0;
  v27 = v19;
LABEL_4:
  result = sub_347A8D0(a1, a2, a3, v9, (__int64)&v27, a5, a7, a6, a8, a9);
  if ( v28 > 0x40 )
  {
    if ( v27 )
    {
      v24 = result;
      j_j___libc_free_0_0(v27);
      return v24;
    }
  }
  return result;
}
