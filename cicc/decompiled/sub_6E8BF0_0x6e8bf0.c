// Function: sub_6E8BF0
// Address: 0x6e8bf0
//
__int64 __fastcall sub_6E8BF0(__int64 a1)
{
  if ( !qword_4D03A68 )
  {
    qword_4D03A68 = sub_724D80(3);
    *(_QWORD *)(qword_4D03A68 + 128) = sub_72C610(2);
    sub_709F60(qword_4D03A68 + 176, 2, 0, 0);
  }
  return sub_6E6A50(qword_4D03A68, a1);
}
