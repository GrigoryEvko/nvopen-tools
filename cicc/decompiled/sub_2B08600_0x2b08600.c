// Function: sub_2B08600
// Address: 0x2b08600
//
__int64 __fastcall sub_2B08600(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a2 + 120);
  if ( !(_DWORD)result )
    return *(unsigned int *)(a2 + 8);
  return result;
}
