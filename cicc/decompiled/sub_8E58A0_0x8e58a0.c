// Function: sub_8E58A0
// Address: 0x8e58a0
//
__int64 __fastcall sub_8E58A0(unsigned __int8 a1, __int64 a2)
{
  __int64 result; // rax

  if ( (unsigned int)a1 - 48 <= 9 )
    return (unsigned int)a1 - 48;
  if ( isxdigit(a1) && islower(a1) )
    return (unsigned int)a1 - 87;
  result = 0;
  if ( !*(_DWORD *)(a2 + 24) )
  {
    ++*(_QWORD *)(a2 + 32);
    ++*(_QWORD *)(a2 + 48);
    *(_DWORD *)(a2 + 24) = 1;
  }
  return result;
}
