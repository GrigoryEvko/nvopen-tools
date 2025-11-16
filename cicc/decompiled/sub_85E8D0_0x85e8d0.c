// Function: sub_85E8D0
// Address: 0x85e8d0
//
__int64 sub_85E8D0()
{
  __int64 i; // rax
  __int64 v1; // rax
  char v2; // dl
  unsigned int v3; // r8d

  for ( i = dword_4F04C64; ; i = *(int *)(v1 + 552) )
  {
    while ( 1 )
    {
      v1 = qword_4F04C68[0] + 776 * i;
      v2 = *(_BYTE *)(v1 + 4);
      if ( v2 != 14 || (*(_BYTE *)(v1 + 12) & 0x10) == 0 )
        break;
      i = *(int *)(v1 + 452);
    }
    v3 = *(_DWORD *)(v1 + 400);
    if ( v3 != -1 )
      break;
    if ( v2 == 6 )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v1 + 208) + 168LL) + 109LL) & 0x20) == 0 )
        return v3;
    }
    else if ( v2 != 1 && v2 != 8 )
    {
      return v3;
    }
  }
  return v3;
}
