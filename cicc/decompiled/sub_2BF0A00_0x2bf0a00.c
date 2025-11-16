// Function: sub_2BF0A00
// Address: 0x2bf0a00
//
__int64 __fastcall sub_2BF0A00(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 48);
  if ( result )
  {
    if ( *(_BYTE *)(result + 128) )
      return *(_QWORD *)(result + 48);
  }
  return result;
}
