// Function: sub_1023590
// Address: 0x1023590
//
__int64 __fastcall sub_1023590(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 32);
  result = 0;
  if ( !*(_WORD *)(v1 + 24) )
    return *(_QWORD *)(v1 + 32);
  return result;
}
