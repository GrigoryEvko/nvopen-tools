// Function: sub_40EDDD
// Address: 0x40eddd
//
__int64 __fastcall sub_40EDDD(__int64 a1, __int64 a2, int a3, const char **a4, __int64 a5)
{
  __int64 result; // rax

  if ( *(_DWORD *)a1 <= 1u )
  {
    sub_130F450(a1, a2);
    if ( *(_DWORD *)a1 <= 1u )
    {
      sub_130F270(a1);
      sub_40EBBB(a1, 2, -1, a3, a4);
      *(_BYTE *)(a1 + 28) = 1;
    }
    return a5;
  }
  return result;
}
