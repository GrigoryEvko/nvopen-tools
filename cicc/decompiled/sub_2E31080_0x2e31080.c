// Function: sub_2E31080
// Address: 0x2e31080
//
__int64 __fastcall sub_2E31080(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12

  result = sub_2E88D60(a2, a2);
  if ( result )
  {
    v3 = result;
    sub_2E78D40(result, a2);
    result = sub_2E866F0(a2, *(_QWORD *)(v3 + 32));
  }
  *(_QWORD *)(a2 + 24) = 0;
  return result;
}
