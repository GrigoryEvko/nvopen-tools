// Function: sub_15E4B00
// Address: 0x15e4b00
//
__int64 __fastcall sub_15E4B00(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( !*(_BYTE *)(a1 + 16) )
    return (*(_DWORD *)(a1 + 32) >> 22) & 1;
  return result;
}
