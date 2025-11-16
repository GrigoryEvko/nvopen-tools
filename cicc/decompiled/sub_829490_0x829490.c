// Function: sub_829490
// Address: 0x829490
//
_BOOL8 __fastcall sub_829490(__int64 a1)
{
  _BOOL8 result; // rax
  __int64 v2; // rcx
  _QWORD *v3; // rdx

  result = 0;
  if ( *(_BYTE *)(a1 + 174) == 7 && (*(_BYTE *)(a1 + 193) & 0x10) != 0 )
  {
    v2 = *(_QWORD *)(a1 + 152);
    v3 = **(_QWORD ***)(v2 + 168);
    if ( v3 )
    {
      if ( !*v3 )
        return *(_QWORD *)(v2 + 160) == v3[1];
    }
  }
  return result;
}
