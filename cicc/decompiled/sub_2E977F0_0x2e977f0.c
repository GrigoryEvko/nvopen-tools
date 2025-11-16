// Function: sub_2E977F0
// Address: 0x2e977f0
//
__int64 __fastcall sub_2E977F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_2EA49A0(a2);
  if ( !result )
  {
    result = sub_2EA48E0(a2);
    if ( result )
      return sub_2E367B0(
               result,
               **(_QWORD **)(a2 + 32),
               *(__int64 **)(a1 + 344),
               *(_QWORD *)(a1 + 352),
               0,
               *(_QWORD *)(a1 + 384));
  }
  return result;
}
