// Function: sub_1318FF0
// Address: 0x1318ff0
//
__int64 sub_1318FF0()
{
  if ( (unsigned __int64)(qword_4C6F228 - 0x4000LL) <= 0x6FFFFFFFFFFFC000LL )
  {
    dword_4F96B5C = sub_1300B70();
    unk_4C6F220 = qword_4C6F228;
    return 1;
  }
  else
  {
    qword_4C6F228 = 0;
    unk_4C6F220 = 0x7000000000001000LL;
    return 0;
  }
}
