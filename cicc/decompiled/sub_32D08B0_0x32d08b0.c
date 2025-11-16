// Function: sub_32D08B0
// Address: 0x32d08b0
//
__int64 __fastcall sub_32D08B0(__int64 a1, __int64 a2, unsigned int a3, int a4)
{
  __int64 v6; // rax
  unsigned __int16 v7; // dx
  __int64 v8; // rax
  __int64 result; // rax
  unsigned int v10; // eax
  unsigned __int64 v11; // rdx
  unsigned __int8 v12; // [rsp+8h] [rbp-58h]
  unsigned __int16 v13; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+18h] [rbp-48h]
  unsigned __int64 v15; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-38h]

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
    v10 = word_4456340[v13 - 1];
    v16 = v10;
    if ( v10 > 0x40 )
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
  v10 = sub_3007130((__int64)&v13, a2);
  v16 = v10;
  if ( v10 > 0x40 )
  {
LABEL_18:
    sub_C43690((__int64)&v15, -1, 1);
    goto LABEL_4;
  }
LABEL_11:
  v11 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v10;
  if ( !v10 )
    v11 = 0;
  v15 = v11;
LABEL_4:
  result = sub_32D0760(a1, a2, a3, a4, (int)&v15, 0);
  if ( v16 > 0x40 )
  {
    if ( v15 )
    {
      v12 = result;
      j_j___libc_free_0_0(v15);
      return v12;
    }
  }
  return result;
}
