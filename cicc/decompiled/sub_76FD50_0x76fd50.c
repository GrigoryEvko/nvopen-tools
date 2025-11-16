// Function: sub_76FD50
// Address: 0x76fd50
//
_QWORD *__fastcall sub_76FD50(_QWORD *a1)
{
  __int64 v1; // rdx
  _QWORD *result; // rax
  __int64 v3; // rdx
  char v4; // cl

  v1 = a1[4];
  for ( result = *(_QWORD **)(v1 + 152); *((_BYTE *)result + 140) == 12; result = (_QWORD *)result[20] )
    ;
  if ( !a1[14] && !a1[12] && !a1[13] && !a1[20] && !a1[29] )
  {
    result = (_QWORD *)result[21];
    v4 = *((_BYTE *)result + 16);
    if ( (v4 & 1) == 0
      && *(char *)(v1 + 204) >= 0
      && (*(_BYTE *)(v1 + 196) & 0x40) == 0
      && ((v4 & 2) != 0 || !*result)
      && !a1[26]
      && (*(_BYTE *)(v1 - 8) & 0x10) == 0
      && (*(_BYTE *)(v1 + 198) & 0x20) == 0 )
    {
      result = &dword_4D045B4;
      if ( !dword_4D045B4 )
      {
        result = &dword_4D04708;
        if ( dword_4D04708 )
          *(_BYTE *)(v1 + 203) |= 2u;
      }
    }
  }
  if ( (*(_BYTE *)(v1 + 203) & 2) == 0 )
  {
    v3 = *(_QWORD *)v1;
    if ( v3 )
      return (_QWORD *)sub_685460(0x2A7u, (FILE *)(v3 + 48), v3);
  }
  return result;
}
