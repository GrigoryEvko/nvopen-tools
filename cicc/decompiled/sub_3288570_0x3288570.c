// Function: sub_3288570
// Address: 0x3288570
//
__int64 __fastcall sub_3288570(__int64 a1, __int64 a2)
{
  unsigned __int16 *v2; // rax
  unsigned __int16 v3; // bx
  __int64 v4; // rax
  int v5; // edx
  unsigned int *v6; // rax
  __int64 v7; // rdx
  unsigned __int16 v9; // [rsp+0h] [rbp-30h] BYREF
  __int64 v10; // [rsp+8h] [rbp-28h]

  v2 = *(unsigned __int16 **)(a1 + 48);
  v3 = *v2;
  v4 = *((_QWORD *)v2 + 1);
  v9 = v3;
  v10 = v4;
  if ( v3 )
  {
    if ( (unsigned __int16)(v3 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v5 = word_4456340[v3 - 1];
  }
  else
  {
    if ( sub_3007100((__int64)&v9) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    v5 = sub_3007130((__int64)&v9, a2);
  }
  v6 = *(unsigned int **)(a1 + 96);
  if ( !v5 )
    return 0;
  v7 = (__int64)&v6[v5 - 1 + 1];
  while ( (*v6 & 0x80000000) != 0 )
  {
    if ( ++v6 == (unsigned int *)v7 )
      return 0;
  }
  return *v6;
}
