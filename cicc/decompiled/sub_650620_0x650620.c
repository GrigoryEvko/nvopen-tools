// Function: sub_650620
// Address: 0x650620
//
unsigned int *__fastcall sub_650620(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int *result; // rax

  if ( !(_DWORD)a3 )
    return sub_6504E0(a1, dword_4F04C64, (__int64 *)&dword_4F063F8, 1, a2, 0);
  sub_863FC0(a1, a2, a3, a4, (unsigned int)a2);
  sub_6504E0(a1, dword_4F04C64, (__int64 *)&dword_4F063F8, 1, a2, 0);
  sub_8602E0(4, a1);
  result = (unsigned int *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  *((_BYTE *)result + 9) |= 0x20u;
  return result;
}
