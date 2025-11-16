// Function: sub_2C2BF80
// Address: 0x2c2bf80
//
__int64 __fastcall sub_2C2BF80(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 i; // rdx

  result = *(_QWORD *)(a1 + 8);
  v2 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = result + 8 * v2; result != i; result += 8 )
  {
    if ( result )
    {
      *(_DWORD *)result = -1;
      *(_BYTE *)(result + 4) = 1;
    }
  }
  return result;
}
