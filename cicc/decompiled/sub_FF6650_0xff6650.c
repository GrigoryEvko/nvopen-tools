// Function: sub_FF6650
// Address: 0xff6650
//
__int64 __fastcall sub_FF6650(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  sub_FF0C10(a1, a2);
  result = *(unsigned int *)(a3 + 8);
  if ( (_DWORD)result )
    return sub_FF5F90(a1, a2, a3);
  return result;
}
