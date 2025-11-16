// Function: sub_1983A80
// Address: 0x1983a80
//
__int64 __fastcall sub_1983A80(__int64 a1, _QWORD *a2)
{
  unsigned __int8 v2; // al
  unsigned int v3; // r8d
  __int64 v4; // rbx
  _QWORD *v5; // rax

  v2 = *(_BYTE *)(a1 + 16);
  v3 = 0;
  if ( v2 <= 0x17u )
    return 0;
  if ( (unsigned int)v2 - 35 <= 0x11 )
  {
    if ( v2 == 35 )
      goto LABEL_4;
    return v3;
  }
  if ( v2 != 56 )
    return v3;
LABEL_4:
  v4 = *(_QWORD *)(a1 + 8);
  if ( v4 )
  {
    while ( 1 )
    {
      v5 = sub_1648700(v4);
      if ( *((_BYTE *)v5 + 16) == 77 && a2 == v5 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        return 0;
    }
    return 1;
  }
  return 0;
}
