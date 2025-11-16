// Function: sub_2FF8060
// Address: 0x2ff8060
//
__int64 __fastcall sub_2FF8060(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_106FD90(*(_QWORD *)(a1 + 192), a2);
  if ( (int)result < 0 )
    return 1000;
  return result;
}
