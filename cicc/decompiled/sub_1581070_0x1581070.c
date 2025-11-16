// Function: sub_1581070
// Address: 0x1581070
//
__int64 __fastcall sub_1581070(unsigned __int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rdx
  __int64 result; // rax
  unsigned int v5; // r12d

  v2 = *(_DWORD *)(a2 + 32);
  if ( v2 > 0x40 )
  {
    v5 = v2 - sub_16A57B0(a2 + 24);
    result = 0;
    if ( v5 > 0x40 )
      return result;
    v3 = **(_QWORD **)(a2 + 24);
  }
  else
  {
    v3 = (__int64)(*(_QWORD *)(a2 + 24) << (64 - (unsigned __int8)v2)) >> (64 - (unsigned __int8)v2);
  }
  result = 0;
  if ( v3 >= 0 )
  {
    LOBYTE(result) = a1 != 0;
    LOBYTE(v3) = v3 >= a1;
    return (unsigned int)v3 & (unsigned int)result ^ 1;
  }
  return result;
}
