// Function: sub_3374B60
// Address: 0x3374b60
//
__int64 __fastcall sub_3374B60(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a2 + 8);
  if ( result == *(_QWORD *)(*(_QWORD *)(a1 + 960) + 8LL) + 320LL )
    return 0;
  return result;
}
