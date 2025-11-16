// Function: sub_2880E50
// Address: 0x2880e50
//
__int64 __fastcall sub_2880E50(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( *(_DWORD *)(a1 + 24) != 2 && !*(_DWORD *)(a1 + 8) )
    return *(unsigned __int8 *)(a1 + 16) ^ 1u;
  return result;
}
