// Function: sub_33CEC10
// Address: 0x33cec10
//
__int64 __fastcall sub_33CEC10(__int64 a1)
{
  __int64 result; // rax
  __int64 i; // rsi
  __int64 v3; // rdx

  result = *(_QWORD *)(a1 + 40);
  for ( i = result + 40LL * *(unsigned int *)(a1 + 64); result != i; *(_DWORD *)(result - 32) = 0 )
  {
    result += 40;
    if ( *(_QWORD *)(result - 40) )
    {
      v3 = *(_QWORD *)(result - 8);
      **(_QWORD **)(result - 16) = v3;
      if ( v3 )
        *(_QWORD *)(v3 + 24) = *(_QWORD *)(result - 16);
    }
    *(_QWORD *)(result - 40) = 0;
  }
  return result;
}
