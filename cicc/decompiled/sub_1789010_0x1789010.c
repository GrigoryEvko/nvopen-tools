// Function: sub_1789010
// Address: 0x1789010
//
__int64 __fastcall sub_1789010(__int64 a1)
{
  __int64 v1; // r12
  char v2; // al
  __int64 result; // rax
  __int64 i; // r13
  _QWORD *v5; // rax
  unsigned __int8 v6; // dl
  __int64 v7; // rax
  char v8; // r8
  __int64 v9; // rdi

  v1 = *(_QWORD *)(a1 - 24);
  v2 = *(_BYTE *)(v1 + 16);
  if ( v2 != 53 )
  {
LABEL_2:
    if ( v2 == 56 )
    {
      v9 = *(_QWORD *)(v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF));
      if ( *(_BYTE *)(v9 + 16) == 53 )
      {
        if ( (unsigned __int8)sub_15F8F00(v9) )
          return (unsigned int)sub_15FA290(v1) ^ 1;
      }
    }
    return 1;
  }
  for ( i = *(_QWORD *)(v1 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v5 = sub_1648700(i);
    v6 = *((_BYTE *)v5 + 16);
    if ( v6 <= 0x17u )
      return 1;
    if ( v6 != 54 )
    {
      if ( v6 != 55 )
        return 1;
      v7 = *(v5 - 3);
      if ( v1 != v7 || !v7 )
        return 1;
    }
  }
  v8 = sub_15F8F00(v1);
  result = 0;
  if ( !v8 )
  {
    v1 = *(_QWORD *)(a1 - 24);
    v2 = *(_BYTE *)(v1 + 16);
    goto LABEL_2;
  }
  return result;
}
