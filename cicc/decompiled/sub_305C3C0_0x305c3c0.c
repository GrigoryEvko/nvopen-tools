// Function: sub_305C3C0
// Address: 0x305c3c0
//
__int64 __fastcall sub_305C3C0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8);
  if ( result == *(_QWORD *)(a1 + 24) )
    return 0;
  return result;
}
