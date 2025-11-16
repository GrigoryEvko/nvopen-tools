// Function: sub_388BEC0
// Address: 0x388bec0
//
__int64 __fastcall sub_388BEC0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  *a2 = 0;
  if ( *(_DWORD *)(a1 + 64) != 45 )
    return 0;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  result = 0;
  *a2 = 1;
  if ( *(_DWORD *)(a1 + 64) == 12 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    result = sub_388BE30(a1, a2);
    if ( !(_BYTE)result )
      return sub_388AF10(a1, 13, "expected ')' after thread local model");
  }
  return result;
}
