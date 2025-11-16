// Function: sub_34494D0
// Address: 0x34494d0
//
__int64 __fastcall sub_34494D0(
        unsigned int *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        _QWORD **a5,
        unsigned int a6,
        __m128i a7)
{
  __int64 v10; // rax
  unsigned __int16 v11; // dx
  __int64 v12; // rax
  __int64 result; // rax
  unsigned int v14; // eax
  _QWORD *v15; // rdx
  __int64 v17; // [rsp+10h] [rbp-60h]
  unsigned __int16 v18; // [rsp+20h] [rbp-50h] BYREF
  __int64 v19; // [rsp+28h] [rbp-48h]
  unsigned __int64 v20; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+38h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v18 = v11;
  v19 = v12;
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 17) > 0x9Eu )
    {
LABEL_3:
      v21 = 1;
      v20 = 1;
      goto LABEL_4;
    }
LABEL_17:
    v14 = word_4456340[v18 - 1];
    v21 = v14;
    if ( v14 > 0x40 )
      goto LABEL_18;
    goto LABEL_11;
  }
  if ( !sub_30070D0((__int64)&v18) )
    goto LABEL_3;
  if ( sub_3007100((__int64)&v18) )
  {
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( v18 )
    {
      if ( (unsigned __int16)(v18 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_17;
    }
  }
  v14 = sub_3007130((__int64)&v18, a2);
  v21 = v14;
  if ( v14 > 0x40 )
  {
LABEL_18:
    sub_C43690((__int64)&v20, -1, 1);
    goto LABEL_4;
  }
LABEL_11:
  v15 = (_QWORD *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v14);
  if ( !v14 )
    v15 = 0;
  v20 = (unsigned __int64)v15;
LABEL_4:
  result = sub_3447D70(a1, a2, a3, a4, (_QWORD **)&v20, a5, a7, a6);
  if ( v21 > 0x40 )
  {
    if ( v20 )
    {
      v17 = result;
      j_j___libc_free_0_0(v20);
      return v17;
    }
  }
  return result;
}
