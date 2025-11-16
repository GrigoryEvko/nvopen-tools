// Function: sub_CB1000
// Address: 0xcb1000
//
__int64 __fastcall sub_CB1000(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 672);
  if ( result )
    return *(_QWORD *)result;
  return result;
}
