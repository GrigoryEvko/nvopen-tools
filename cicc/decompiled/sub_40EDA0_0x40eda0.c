// Function: sub_40EDA0
// Address: 0x40eda0
//
__int64 __fastcall sub_40EDA0(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // edx
  int v4; // ecx
  int v5; // r8d
  int v6; // r9d
  __int64 result; // rax

  if ( *(_DWORD *)a1 <= 1u )
  {
    sub_130F450(a1, a2);
    if ( *(_DWORD *)a1 <= 1u )
    {
      sub_130F270(a1);
      sub_130F0B0(a1, (unsigned int)"[", v3, v4, v5, v6);
      ++*(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 28) = 0;
    }
    return a3;
  }
  return result;
}
