// Function: sub_12FA770
// Address: 0x12fa770
//
__int64 __fastcall sub_12FA770(__int64 a1, unsigned __int16 a2)
{
  __int64 v2; // rbp
  _QWORD v4[7]; // [rsp-38h] [rbp-38h] BYREF

  if ( (a2 & 0x7FFF) != 0x7FFF || (a1 & 0x7FFFFFFFFFFFFFFFLL) == 0 )
    return a1 << 49;
  v4[6] = v2;
  sub_12FB9C0(a2, a1, v4);
  return sub_12FBB00(v4);
}
