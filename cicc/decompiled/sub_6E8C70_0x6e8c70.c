// Function: sub_6E8C70
// Address: 0x6e8c70
//
__int64 __fastcall sub_6E8C70(__int64 a1)
{
  if ( !qword_4D03A60 )
  {
    qword_4D03A60 = sub_724D80(3);
    *(_QWORD *)(qword_4D03A60 + 128) = sub_72C610(2);
    sub_70A170(qword_4D03A60 + 176, 2);
  }
  return sub_6E6A50(qword_4D03A60, a1);
}
