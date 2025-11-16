// Function: sub_B73450
// Address: 0xb73450
//
__int64 __fastcall sub_B73450(__int64 a1)
{
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 8) & 1) == 0 )
    return sub_C7D6A0(*(_QWORD *)(a1 + 16), 24LL * *(unsigned int *)(a1 + 24), 8);
  return result;
}
