// Function: sub_85BC50
// Address: 0x85bc50
//
__int64 __fastcall sub_85BC50(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 result; // rax
  __int64 v4; // rsi
  __int64 v5; // [rsp+0h] [rbp-20h] BYREF
  _QWORD v6[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = a1;
  v5 = a2;
  for ( v6[0] = a1; v2; v6[0] = v2 )
  {
    *(_BYTE *)(v2[1] + 82LL) |= 1u;
    v2 = (_QWORD *)*v2;
  }
  sub_89ED70(a1, a2, v6, &v5);
  result = v6[0];
  while ( v6[0] )
  {
    v4 = v5;
    if ( !v5 )
      return result;
    if ( *(_BYTE *)(v5 + 8) != 3 )
      goto LABEL_5;
    v4 = *(_QWORD *)v5;
    if ( *(_QWORD *)v5 && ((*(_BYTE *)(v4 + 24) & 8) != 0 || (*(_BYTE *)(result + 56) & 0x10) == 0) )
    {
      if ( *(_BYTE *)(v4 + 8) != 3 )
      {
        v5 = *(_QWORD *)v5;
LABEL_5:
        if ( !result )
          return result;
        sub_85BC00(*(_QWORD *)(result + 8), v4);
        result = v6[0];
        goto LABEL_7;
      }
      if ( !result )
        return result;
    }
    else
    {
      sub_85B8C0(*(_QWORD *)(result + 8), v4);
      result = v6[0];
      if ( !v6[0] )
        return result;
    }
LABEL_7:
    *(_BYTE *)(*(_QWORD *)(result + 8) + 82LL) &= ~1u;
    sub_89ED80(v6, &v5);
    result = v6[0];
  }
  return result;
}
