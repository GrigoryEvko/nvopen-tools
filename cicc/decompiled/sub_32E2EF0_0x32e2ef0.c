// Function: sub_32E2EF0
// Address: 0x32e2ef0
//
__int64 __fastcall sub_32E2EF0(__int64 a1, __int64 a2, int a3)
{
  unsigned __int16 *v4; // rdx
  int v5; // eax
  __int64 v6; // rdx
  __int64 result; // rax
  unsigned int v8; // eax
  unsigned __int64 v9; // rdx
  unsigned __int8 v10; // [rsp+Fh] [rbp-31h]
  unsigned __int64 v11; // [rsp+10h] [rbp-30h] BYREF
  __int64 v12; // [rsp+18h] [rbp-28h]

  v4 = *(unsigned __int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  LOWORD(v11) = v5;
  v12 = v6;
  if ( (_WORD)v5 )
  {
    if ( (unsigned __int16)(v5 - 176) <= 0x34u )
      return 0;
    goto LABEL_15;
  }
  if ( sub_3007100((__int64)&v11) )
    return 0;
  if ( sub_3007100((__int64)&v11) )
  {
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    v5 = (unsigned __int16)v11;
    if ( (_WORD)v11 )
    {
LABEL_15:
      if ( (unsigned __int16)(v5 - 176) <= 0x34u )
      {
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
        v5 = (unsigned __int16)v11;
      }
      v8 = word_4456340[v5 - 1];
      LODWORD(v12) = v8;
      if ( v8 > 0x40 )
        goto LABEL_18;
LABEL_8:
      v9 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v8;
      if ( !v8 )
        v9 = 0;
      v11 = v9;
      goto LABEL_11;
    }
  }
  v8 = sub_3007130((__int64)&v11, a2);
  LODWORD(v12) = v8;
  if ( v8 <= 0x40 )
    goto LABEL_8;
LABEL_18:
  sub_C43690((__int64)&v11, -1, 1);
LABEL_11:
  result = sub_32E2DA0(a1, a2, a3, (int)&v11, 0);
  if ( (unsigned int)v12 > 0x40 )
  {
    if ( v11 )
    {
      v10 = result;
      j_j___libc_free_0_0(v11);
      return v10;
    }
  }
  return result;
}
