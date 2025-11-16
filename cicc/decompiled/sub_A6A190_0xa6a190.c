// Function: sub_A6A190
// Address: 0xa6a190
//
__int64 __fastcall sub_A6A190(__int64 a1)
{
  __int64 result; // rax

  if ( !*(_QWORD *)(a1 + 96) )
    return 0;
  result = sub_A69D70(a1);
  *(_QWORD *)(a1 + 96) = 0;
  return result;
}
