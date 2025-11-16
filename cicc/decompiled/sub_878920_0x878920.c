// Function: sub_878920
// Address: 0x878920
//
__int64 __fastcall sub_878920(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 96);
  result = *(_QWORD *)(v1 + 72);
  if ( !result )
    return *(_QWORD *)(v1 + 104);
  return result;
}
