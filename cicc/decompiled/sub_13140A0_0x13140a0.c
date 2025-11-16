// Function: sub_13140A0
// Address: 0x13140a0
//
int __fastcall sub_13140A0(__int64 a1)
{
  int result; // eax
  bool v2; // cc

  result = (unsigned int)sub_130B060(a1, qword_4F96AE0);
  v2 = *(_BYTE *)(a1 + 816) <= 2u;
  qword_4F96B50 = 0;
  if ( v2 )
    return sub_1313920(a1);
  return result;
}
