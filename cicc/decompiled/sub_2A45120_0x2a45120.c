// Function: sub_2A45120
// Address: 0x2a45120
//
__int64 __fastcall sub_2A45120(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  result = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)result )
  {
    do
    {
      result = sub_2A45070(a1, a2, a3);
      if ( (_BYTE)result )
        break;
    }
    while ( (*(_DWORD *)(a2 + 8))-- != 1 );
  }
  return result;
}
