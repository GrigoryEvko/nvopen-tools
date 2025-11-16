// Function: sub_3177070
// Address: 0x3177070
//
__int64 __fastcall sub_3177070(__int64 **a1, __int64 a2, _BYTE *a3)
{
  __int64 v3; // rax
  _BYTE *v4; // rsi
  _BYTE *v5; // rcx

  v3 = *(_QWORD *)(a2 + 16);
  if ( !v3 )
    return 0;
  v4 = 0;
  do
  {
    v5 = *(_BYTE **)(v3 + 24);
    if ( a3 != v5 )
    {
      if ( *v5 != 62 || v4 || (v5[2] & 1) != 0 )
        return 0;
      v4 = (_BYTE *)*((_QWORD *)v5 - 8);
    }
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v3 );
  if ( v4 )
    return sub_3176FF0(a1, v4);
  else
    return 0;
}
