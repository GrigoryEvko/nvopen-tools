// Function: sub_1B2B4D0
// Address: 0x1b2b4d0
//
__int64 __fastcall sub_1B2B4D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  result = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)result )
  {
    do
    {
      result = sub_1B2B3A0(a1, a2, a3);
      if ( (_BYTE)result )
        break;
    }
    while ( (*(_DWORD *)(a2 + 8))-- != 1 );
  }
  return result;
}
