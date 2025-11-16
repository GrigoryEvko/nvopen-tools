// Function: sub_30368A0
// Address: 0x30368a0
//
__int64 __fastcall sub_30368A0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  unsigned int v8; // r13d
  __int64 v9; // rax
  _QWORD v10[6]; // [rsp+0h] [rbp-30h] BYREF

  v10[0] = a4;
  v10[1] = a5;
  if ( (_WORD)a4 )
  {
    if ( (unsigned __int16)(a4 - 17) > 0xD3u )
      return 2;
    if ( (unsigned __int16)(a4 - 176) > 0x34u )
      goto LABEL_11;
  }
  else
  {
    if ( !sub_30070B0((__int64)v10) )
      return 2;
    if ( !sub_3007100((__int64)v10) )
      goto LABEL_6;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( LOWORD(v10[0]) )
  {
    if ( (unsigned __int16)(LOWORD(v10[0]) - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_11:
    v8 = word_4456340[LOWORD(v10[0]) - 1];
    goto LABEL_7;
  }
LABEL_6:
  v8 = sub_3007130((__int64)v10, a2);
LABEL_7:
  LOWORD(v9) = sub_2D43050(2, v8);
  if ( !(_WORD)v9 )
  {
    v9 = sub_3009400(a3, 2, 0, v8, 0);
    v5 = v9;
  }
  LOWORD(v5) = v9;
  return v5;
}
