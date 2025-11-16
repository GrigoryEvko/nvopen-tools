// Function: sub_7AD570
// Address: 0x7ad570
//
_BOOL8 __fastcall sub_7AD570(__int64 a1, __int64 a2)
{
  char v2; // dl
  __int64 v3; // rax
  _BOOL8 result; // rax
  __int64 v5; // rcx
  __int64 v6; // r8

  v2 = *(_BYTE *)(a2 + 140);
  if ( v2 == 12 )
  {
    v3 = a2;
    do
    {
      v3 = *(_QWORD *)(v3 + 160);
      v2 = *(_BYTE *)(v3 + 140);
    }
    while ( v2 == 12 );
  }
  result = 0;
  if ( v2 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u && *(_QWORD *)(*(_QWORD *)(a1 + 168) + 256LL)
      || (unsigned int)sub_8D3D40(a2) )
    {
      return 1;
    }
    else
    {
      result = 1;
      if ( a2 != a1 )
        return (unsigned int)sub_8D97D0(a1, a2, 0, v5, v6) != 0;
    }
  }
  return result;
}
