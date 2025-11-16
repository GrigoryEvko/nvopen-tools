// Function: sub_7D0530
// Address: 0x7d0530
//
__int64 __fastcall sub_7D0530(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 8LL);
  if ( !result )
    return a1;
  return result;
}
