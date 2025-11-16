// Function: sub_2B3F3B0
// Address: 0x2b3f3b0
//
__int64 __fastcall sub_2B3F3B0(__int64 a1)
{
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    return sub_C7D6A0(*(_QWORD *)(a1 + 16), 8LL * *(unsigned int *)(a1 + 24), 8);
  return result;
}
