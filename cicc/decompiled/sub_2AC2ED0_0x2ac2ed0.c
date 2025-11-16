// Function: sub_2AC2ED0
// Address: 0x2ac2ed0
//
__int64 __fastcall sub_2AC2ED0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  __int64 i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = result + 72 * v1; result != i; result += 72 )
  {
    if ( result )
    {
      *(_DWORD *)result = -1;
      *(_BYTE *)(result + 4) = 1;
    }
  }
  return result;
}
