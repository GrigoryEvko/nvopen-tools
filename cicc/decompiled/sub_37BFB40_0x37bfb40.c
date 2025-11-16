// Function: sub_37BFB40
// Address: 0x37bfb40
//
__int64 __fastcall sub_37BFB40(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  __int64 i; // rdx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = result + 32 * v1; result != i; result += 32 )
  {
    if ( result )
    {
      *(_QWORD *)result = -4096;
      *(_DWORD *)(result + 8) = -1;
    }
  }
  return result;
}
