// Function: sub_108B080
// Address: 0x108b080
//
unsigned __int64 __fastcall sub_108B080(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax

  result = *(_QWORD *)(a1 + 24) + a3;
  *(_QWORD *)(a1 + 32) = a3;
  if ( result > a2 )
    sub_108B060(a1);
  return result;
}
