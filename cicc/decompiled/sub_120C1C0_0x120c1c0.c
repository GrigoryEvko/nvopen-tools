// Function: sub_120C1C0
// Address: 0x120c1c0
//
__int64 __fastcall sub_120C1C0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  *a2 = 0;
  if ( *(_DWORD *)(a1 + 240) != 47 )
    return 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  result = 0;
  *a2 = 1;
  if ( *(_DWORD *)(a1 + 240) == 12 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    result = sub_120C120(a1, a2);
    if ( !(_BYTE)result )
      return sub_120AFE0(a1, 13, "expected ')' after thread local model");
  }
  return result;
}
