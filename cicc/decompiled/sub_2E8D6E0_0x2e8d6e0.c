// Function: sub_2E8D6E0
// Address: 0x2e8d6e0
//
__int64 __fastcall sub_2E8D6E0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 result; // rax
  __int64 i; // r12
  unsigned int v7; // edx

  v4 = *(_QWORD *)(a1 + 32);
  if ( a2 - 1 >= 0x3FFFFFFF )
    a3 = 0;
  result = 5LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF);
  for ( i = v4 + 40LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF); i != v4; v4 += 40 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)v4 )
        goto LABEL_11;
      result = *(unsigned __int8 *)(v4 + 3);
      if ( (result & 0x10) != 0 || (result & 0x40) == 0 )
        goto LABEL_11;
      v7 = *(_DWORD *)(v4 + 8);
      if ( a3 )
        break;
      if ( v7 != a2 )
        goto LABEL_11;
LABEL_14:
      *(_BYTE *)(v4 + 3) &= ~0x40u;
LABEL_15:
      v4 += 40;
      if ( i == v4 )
        return result;
    }
    if ( v7 == a2 )
      goto LABEL_14;
    if ( a2 - 1 <= 0x3FFFFFFE )
    {
      result = v7 - 1;
      if ( (unsigned int)result <= 0x3FFFFFFE )
      {
        result = sub_E92070(a3, a2, v7);
        if ( (_BYTE)result )
        {
          *(_BYTE *)(v4 + 3) &= ~0x40u;
          goto LABEL_15;
        }
      }
    }
LABEL_11:
    ;
  }
  return result;
}
