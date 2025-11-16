// Function: sub_2AC3110
// Address: 0x2ac3110
//
__int64 __fastcall sub_2AC3110(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  __int64 i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = result + 40 * v1; result != i; result += 40 )
  {
    if ( result )
    {
      *(_QWORD *)result = -4096;
      *(_DWORD *)(result + 8) = -1;
      *(_BYTE *)(result + 12) = 1;
    }
  }
  return result;
}
