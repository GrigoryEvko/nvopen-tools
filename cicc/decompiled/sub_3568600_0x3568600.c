// Function: sub_3568600
// Address: 0x3568600
//
__int64 __fastcall sub_3568600(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  do
  {
    result = a2;
    a2 = *(_QWORD *)(a2 + 8);
  }
  while ( a2 );
  return result;
}
