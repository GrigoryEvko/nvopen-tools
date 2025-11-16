// Function: sub_1E15BB0
// Address: 0x1e15bb0
//
__int64 __fastcall sub_1E15BB0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 24);
  if ( result )
    return *(_QWORD *)(*(_QWORD *)(result + 56) + 40LL);
  return result;
}
