// Function: sub_1776740
// Address: 0x1776740
//
__int64 __fastcall sub_1776740(__int64 a1)
{
  __int64 v1; // rdi
  unsigned __int8 v2; // al
  __int64 v4; // rax
  int v5; // eax

  v1 = sub_1649C60(a1);
  v2 = *(_BYTE *)(v1 + 16);
  if ( v2 > 0x17u )
  {
    while ( v2 == 78 )
    {
      v4 = *(_QWORD *)(v1 - 24);
      if ( *(_BYTE *)(v4 + 16) )
        break;
      if ( (*(_BYTE *)(v4 + 33) & 0x20) == 0 )
        break;
      v5 = *(_DWORD *)(v4 + 36);
      if ( v5 != 4046 && v5 != 4242 )
        break;
      v1 = sub_1649C60(*(_QWORD *)(v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF)));
      v2 = *(_BYTE *)(v1 + 16);
      if ( v2 <= 0x17u )
        goto LABEL_2;
    }
  }
  else
  {
LABEL_2:
    if ( v2 == 17 )
      return sub_15E0450(v1);
  }
  return 0;
}
