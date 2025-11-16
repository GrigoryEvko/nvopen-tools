// Function: sub_33DE0A0
// Address: 0x33de0a0
//
__int64 __fastcall sub_33DE0A0(
        _QWORD **a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        unsigned __int8 a5,
        unsigned int a6)
{
  __int64 v9; // rax
  unsigned __int16 v10; // dx
  __int64 v11; // rax
  __int64 result; // rax
  unsigned int v13; // eax
  unsigned __int64 v14; // rdx
  unsigned __int8 v16; // [rsp+Ch] [rbp-54h]
  unsigned __int16 v17; // [rsp+10h] [rbp-50h] BYREF
  __int64 v18; // [rsp+18h] [rbp-48h]
  unsigned __int64 v19; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v10 = *(_WORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  v17 = v10;
  v18 = v11;
  if ( v10 )
  {
    if ( (unsigned __int16)(v10 - 17) > 0x9Eu )
    {
LABEL_3:
      v20 = 1;
      v19 = 1;
      goto LABEL_4;
    }
LABEL_17:
    v13 = word_4456340[v17 - 1];
    v20 = v13;
    if ( v13 > 0x40 )
      goto LABEL_18;
    goto LABEL_11;
  }
  if ( !sub_30070D0((__int64)&v17) )
    goto LABEL_3;
  if ( sub_3007100((__int64)&v17) )
  {
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( v17 )
    {
      if ( (unsigned __int16)(v17 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_17;
    }
  }
  v13 = sub_3007130((__int64)&v17, a2);
  v20 = v13;
  if ( v13 > 0x40 )
  {
LABEL_18:
    sub_C43690((__int64)&v19, -1, 1);
    goto LABEL_4;
  }
LABEL_11:
  v14 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
  if ( !v13 )
    v14 = 0;
  v19 = v14;
LABEL_4:
  result = sub_33DDBD0(a1, a2, a3, (__int64)&v19, a4, a5, a6);
  if ( v20 > 0x40 )
  {
    if ( v19 )
    {
      v16 = result;
      j_j___libc_free_0_0(v19);
      return v16;
    }
  }
  return result;
}
