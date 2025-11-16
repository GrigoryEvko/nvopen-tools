// Function: sub_1E1A610
// Address: 0x1e1a610
//
__int64 __fastcall sub_1E1A610(__int64 a1, int a2)
{
  __int64 result; // rax
  __int64 i; // rcx
  char v4; // dl

  result = *(_QWORD *)(a1 + 32);
  for ( i = result + 40LL * *(unsigned int *)(a1 + 40); i != result; result += 40 )
  {
    if ( !*(_BYTE *)result )
    {
      v4 = *(_BYTE *)(result + 3);
      if ( (v4 & 0x10) != 0 && a2 == *(_DWORD *)(result + 8) )
        *(_BYTE *)(result + 3) = v4 & 0xBF;
    }
  }
  return result;
}
