// Function: sub_5F0640
// Address: 0x5f0640
//
__int64 __fastcall sub_5F0640(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax
  __int64 result; // rax
  _BYTE *v4; // rdx

  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  v1 = *(_QWORD *)(a1 + 168);
  v2 = *(_QWORD *)(*(_QWORD *)(v1 + 152) + 144LL);
  if ( !v2 )
  {
LABEL_11:
    result = *(_QWORD *)(v1 + 136);
    if ( !result )
      return 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v4 = *(_BYTE **)(result + 8);
        if ( (v4[206] & 8) != 0 )
          break;
LABEL_14:
        result = *(_QWORD *)result;
        if ( !result )
          return result;
      }
      if ( v4[174] == 5 )
      {
        if ( v4[176] == 30 )
          return 1;
        goto LABEL_14;
      }
      result = *(_QWORD *)result;
      if ( !result )
        return result;
    }
  }
  while ( (*(_BYTE *)(v2 + 206) & 8) == 0 || *(_BYTE *)(v2 + 174) != 5 || *(_BYTE *)(v2 + 176) != 30 )
  {
    v2 = *(_QWORD *)(v2 + 112);
    if ( !v2 )
      goto LABEL_11;
  }
  return 1;
}
