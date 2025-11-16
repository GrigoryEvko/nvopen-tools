// Function: sub_33DD090
// Address: 0x33dd090
//
__int64 __fastcall sub_33DD090(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v8; // rax
  unsigned __int16 v9; // dx
  __int64 v10; // rax
  unsigned int v12; // eax
  unsigned __int64 v13; // rdx
  unsigned __int16 v14; // [rsp+10h] [rbp-50h] BYREF
  __int64 v15; // [rsp+18h] [rbp-48h]
  unsigned __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-38h]

  v8 = *(_QWORD *)(a3 + 48) + 16LL * (unsigned int)a4;
  v9 = *(_WORD *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v14 = v9;
  v15 = v10;
  if ( v9 )
  {
    if ( (unsigned __int16)(v9 - 17) > 0x9Eu )
    {
LABEL_3:
      v17 = 1;
      v16 = 1;
      goto LABEL_4;
    }
LABEL_17:
    v12 = word_4456340[v14 - 1];
    v17 = v12;
    if ( v12 > 0x40 )
      goto LABEL_18;
    goto LABEL_11;
  }
  if ( !sub_30070D0((__int64)&v14) )
    goto LABEL_3;
  if ( sub_3007100((__int64)&v14) )
  {
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( v14 )
    {
      if ( (unsigned __int16)(v14 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_17;
    }
  }
  v12 = sub_3007130((__int64)&v14, a2);
  v17 = v12;
  if ( v12 > 0x40 )
  {
LABEL_18:
    sub_C43690((__int64)&v16, -1, 1);
    goto LABEL_4;
  }
LABEL_11:
  v13 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v12;
  if ( !v12 )
    v13 = 0;
  v16 = v13;
LABEL_4:
  sub_33D4EF0(a1, a2, a3, a4, (__int64)&v16, a5);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  return a1;
}
