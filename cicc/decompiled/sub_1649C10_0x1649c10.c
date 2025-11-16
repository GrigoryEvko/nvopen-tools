// Function: sub_1649C10
// Address: 0x1649c10
//
__int64 __fastcall sub_1649C10(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 24);
  if ( result )
  {
    if ( result != -8 && result != -16 )
      result = sub_1649B30((_QWORD *)(a1 + 8));
    *(_QWORD *)(a1 + 24) = 0;
  }
  return result;
}
