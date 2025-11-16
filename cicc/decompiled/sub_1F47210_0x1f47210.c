// Function: sub_1F47210
// Address: 0x1f47210
//
__int64 __fastcall sub_1F47210(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax

  sub_1F46F00(a1, &unk_4FC8A0C, 0, 1, 0);
  result = sub_1F46F00(a1, &unk_4FCE24C, 0, 1, 0);
  if ( a2 )
    return sub_1F46490(a1, a2, 1, 1, 0);
  return result;
}
