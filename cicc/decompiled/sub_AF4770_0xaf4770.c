// Function: sub_AF4770
// Address: 0xaf4770
//
bool __fastcall sub_AF4770(__int64 a1)
{
  bool result; // al
  _QWORD v2[2]; // [rsp+0h] [rbp-20h] BYREF
  bool v3; // [rsp+10h] [rbp-10h]

  sub_AF4640((__int64)v2, a1);
  result = v3;
  if ( v3 )
  {
    result = 0;
    if ( v2[1] == 1 )
      return *(_QWORD *)v2[0] == 6;
  }
  return result;
}
