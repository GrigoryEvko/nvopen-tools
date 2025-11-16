// Function: sub_1B16970
// Address: 0x1b16970
//
__int64 __fastcall sub_1B16970(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 32);
  result = 0;
  if ( !*(_WORD *)(v1 + 24) )
    return *(_QWORD *)(v1 + 32);
  return result;
}
