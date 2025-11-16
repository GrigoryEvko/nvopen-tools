// Function: sub_341E370
// Address: 0x341e370
//
__int64 __fastcall sub_341E370(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( a2 )
    a2 += 8;
  result = *(_QWORD *)(a1 + 24);
  if ( *(_QWORD *)result == a2 )
    *(_QWORD *)result = *(_QWORD *)(*(_QWORD *)result + 8LL);
  return result;
}
