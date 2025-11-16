// Function: sub_33DCDA0
// Address: 0x33dcda0
//
const void **__fastcall sub_33DCDA0(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  __int64 v6; // rax
  unsigned __int16 v7; // dx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v11; // eax
  unsigned __int64 v12; // rdx
  unsigned __int16 v13; // [rsp+20h] [rbp-60h] BYREF
  __int64 v14; // [rsp+28h] [rbp-58h]
  unsigned __int64 v15; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+38h] [rbp-48h]
  const void **v17; // [rsp+40h] [rbp-40h]
  __int64 v18; // [rsp+48h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v7 = *(_WORD *)v6;
  v8 = *(_QWORD *)(v6 + 8);
  v13 = v7;
  v14 = v8;
  if ( v7 )
  {
    if ( (unsigned __int16)(v7 - 17) > 0x9Eu )
    {
LABEL_3:
      v16 = 1;
      v15 = 1;
      goto LABEL_4;
    }
LABEL_17:
    v11 = word_4456340[v13 - 1];
    v16 = v11;
    if ( v11 > 0x40 )
      goto LABEL_18;
    goto LABEL_11;
  }
  if ( !sub_30070D0((__int64)&v13) )
    goto LABEL_3;
  if ( sub_3007100((__int64)&v13) )
  {
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( v13 )
    {
      if ( (unsigned __int16)(v13 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_17;
    }
  }
  v11 = sub_3007130((__int64)&v13, a2);
  v16 = v11;
  if ( v11 > 0x40 )
  {
LABEL_18:
    sub_C43690((__int64)&v15, -1, 1);
    goto LABEL_4;
  }
LABEL_11:
  v12 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
  if ( !v11 )
    v12 = 0;
  v15 = v12;
LABEL_4:
  v17 = sub_33DCC00(a1, a2, a3, (__int64)&v15, a4);
  v18 = v9;
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  return v17;
}
