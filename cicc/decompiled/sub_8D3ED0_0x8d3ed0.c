// Function: sub_8D3ED0
// Address: 0x8d3ed0
//
_BOOL8 __fastcall sub_8D3ED0(__int64 a1)
{
  _BOOL8 result; // rax

  result = sub_8D3EA0(a1);
  if ( result )
    return *(_DWORD *)(*(_QWORD *)(a1 + 168) + 24LL) == 2;
  return result;
}
