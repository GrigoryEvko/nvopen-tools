// Function: sub_1301030
// Address: 0x1301030
//
__int64 __fastcall sub_1301030(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 result; // rax

  if ( a2 > 0xFFE )
    return 0;
  if ( (unsigned int)sub_1300B70() == a2 )
    _InterlockedAdd(&dword_4F96988, 1u);
  result = qword_50579C0[a2];
  if ( !result )
    return sub_1317810(a1, a2, a3);
  return result;
}
