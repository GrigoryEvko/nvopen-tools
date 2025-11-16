// Function: sub_3542A80
// Address: 0x3542a80
//
__int64 __fastcall sub_3542A80(__int64 a1, int a2)
{
  __int64 result; // rax

  result = 0;
  if ( !*(_DWORD *)(a1 + 572) )
  {
    result = 1;
    if ( dword_503DD68 != 2 )
    {
      LOBYTE(result) = dword_503DD68 != 1;
      return (a2 | (unsigned int)result) ^ 1;
    }
  }
  return result;
}
