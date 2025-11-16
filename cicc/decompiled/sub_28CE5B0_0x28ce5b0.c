// Function: sub_28CE5B0
// Address: 0x28ce5b0
//
__int64 __fastcall sub_28CE5B0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  __int64 i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = result + 24 * v1; result != i; result += 24 )
  {
    if ( result )
    {
      *(_QWORD *)result = -4096;
      *(_DWORD *)(result + 8) = -1;
    }
  }
  return result;
}
