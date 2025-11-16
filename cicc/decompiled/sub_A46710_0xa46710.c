// Function: sub_A46710
// Address: 0xa46710
//
__int64 __fastcall sub_A46710(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  __int64 result; // rax

  for ( i = *(_QWORD *)(a2 + 80); a2 + 72 != i; i = *(_QWORD *)(i + 8) )
    result = sub_A466C0(a1, i);
  return result;
}
