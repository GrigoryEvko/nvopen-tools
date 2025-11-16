// Function: sub_6E8F10
// Address: 0x6e8f10
//
__int64 __fastcall sub_6E8F10(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 i; // r12
  __int64 result; // rax
  char v8; // dl
  char v9; // cl

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  for ( i = *(_QWORD *)(a1 + 160); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (unsigned int)sub_8D3D40(i) )
    return 1;
  result = sub_8D3D40(a2);
  if ( (_DWORD)result )
    return 1;
  v8 = *(_BYTE *)(i + 140);
  if ( (unsigned __int8)(v8 - 2) <= 1u )
  {
    v9 = *(_BYTE *)(a2 + 140);
    if ( v8 == v9 )
    {
      if ( v9 != 2 )
      {
LABEL_13:
        if ( !a4 )
          return (unsigned int)sub_8D67E0(a2, a3, i, 0, 0) == 0;
        return 1;
      }
    }
    else if ( v9 != 2 )
    {
      return result;
    }
    if ( (*(_BYTE *)(a2 + 161) & 8) == 0 )
      goto LABEL_13;
  }
  return result;
}
