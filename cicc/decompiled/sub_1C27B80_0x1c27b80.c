// Function: sub_1C27B80
// Address: 0x1c27b80
//
__int64 __fastcall sub_1C27B80(__int64 a1, unsigned int a2)
{
  if ( !(_BYTE)a2 )
    return 0;
  if ( sub_22416F0(a1, "i128", 0, 4) != -1 )
    return 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8)) <= 0xC )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(a1, "-i128:128:128", 13);
  return a2;
}
