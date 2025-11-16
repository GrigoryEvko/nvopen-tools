// Function: sub_1E1A650
// Address: 0x1e1a650
//
__int64 __fastcall sub_1E1A650(__int64 a1, int a2, char a3)
{
  __int64 result; // rax
  __int64 v6; // rdx
  char v7; // di

  result = *(_QWORD *)(a1 + 32);
  v6 = result + 40LL * *(unsigned int *)(a1 + 40);
  if ( result != v6 )
  {
    v7 = a3 & 1;
    do
    {
      if ( !*(_BYTE *)result
        && (*(_BYTE *)(result + 3) & 0x10) != 0
        && a2 == *(_DWORD *)(result + 8)
        && (*(_DWORD *)result & 0xFFF00) != 0 )
      {
        *(_BYTE *)(result + 4) = v7 | *(_BYTE *)(result + 4) & 0xFE;
      }
      result += 40;
    }
    while ( v6 != result );
  }
  return result;
}
