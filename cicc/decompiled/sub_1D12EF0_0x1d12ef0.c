// Function: sub_1D12EF0
// Address: 0x1d12ef0
//
__int64 __fastcall sub_1D12EF0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 result; // rax

  v2 = *(__int64 **)(a1 + 24);
  for ( result = *v2; **(_QWORD **)(a1 + 32) != *v2; result = *v2 )
  {
    if ( a2 != *(_QWORD *)(result + 16) )
      break;
    *v2 = *(_QWORD *)(result + 32);
    v2 = *(__int64 **)(a1 + 24);
  }
  return result;
}
