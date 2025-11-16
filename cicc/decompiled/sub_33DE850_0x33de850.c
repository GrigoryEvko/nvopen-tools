// Function: sub_33DE850
// Address: 0x33de850
//
__int64 __fastcall sub_33DE850(_QWORD **a1, __int64 a2, __int64 a3, unsigned __int8 a4, unsigned int a5)
{
  __int64 result; // rax
  __int64 v9; // rax
  unsigned __int16 v10; // dx
  __int64 v11; // rax
  unsigned int v12; // eax
  unsigned __int64 v13; // rdx
  unsigned __int8 v14; // [rsp-60h] [rbp-60h]
  unsigned __int16 v15; // [rsp-58h] [rbp-58h] BYREF
  __int64 v16; // [rsp-50h] [rbp-50h]
  unsigned __int64 v17; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v18; // [rsp-40h] [rbp-40h]

  result = 1;
  if ( *(_DWORD *)(a2 + 24) == 52 )
    return result;
  v9 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v10 = *(_WORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  v15 = v10;
  v16 = v11;
  if ( v10 )
  {
    if ( (unsigned __int16)(v10 - 17) > 0x9Eu )
      goto LABEL_4;
    goto LABEL_10;
  }
  if ( sub_30070D0((__int64)&v15) )
  {
    if ( !sub_3007100((__int64)&v15)
      || (sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead"),
          !v15) )
    {
      v12 = sub_3007130((__int64)&v15, a2);
LABEL_11:
      v18 = v12;
      if ( v12 > 0x40 )
      {
        sub_C43690((__int64)&v17, -1, 1);
      }
      else
      {
        v13 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v12;
        if ( !v12 )
          v13 = 0;
        v17 = v13;
      }
      goto LABEL_5;
    }
    if ( (unsigned __int16)(v15 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_10:
    v12 = word_4456340[v15 - 1];
    goto LABEL_11;
  }
LABEL_4:
  v18 = 1;
  v17 = 1;
LABEL_5:
  result = sub_33DE230(a1, a2, a3, (__int64)&v17, a4, a5);
  if ( v18 > 0x40 )
  {
    if ( v17 )
    {
      v14 = result;
      j_j___libc_free_0_0(v17);
      return v14;
    }
  }
  return result;
}
