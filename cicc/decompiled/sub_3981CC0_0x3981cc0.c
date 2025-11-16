// Function: sub_3981CC0
// Address: 0x3981cc0
//
unsigned __int64 __fastcall sub_3981CC0(__int64 a1)
{
  unsigned __int64 result; // rax

  result = *(_QWORD *)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*(_QWORD *)(a1 + 40) & 4) != 0 )
    return 0;
  return result;
}
