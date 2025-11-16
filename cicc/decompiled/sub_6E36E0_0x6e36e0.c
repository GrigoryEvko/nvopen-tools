// Function: sub_6E36E0
// Address: 0x6e36e0
//
__int64 __fastcall sub_6E36E0(__int64 a1)
{
  __int64 result; // rax

  result = a1;
  if ( *(_BYTE *)(a1 + 24) == 1 && *(_BYTE *)(a1 + 56) == 4 )
    return *(_QWORD *)(a1 + 72);
  return result;
}
