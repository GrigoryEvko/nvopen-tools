// Function: sub_37BFF30
// Address: 0x37bff30
//
__int64 __fastcall sub_37BFF30(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  __int64 i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = result + 48 * v1; result != i; result += 48 )
  {
    if ( result )
    {
      *(_QWORD *)result = 0;
      *(_BYTE *)(result + 24) = 0;
      *(_QWORD *)(result + 32) = 0;
    }
  }
  return result;
}
