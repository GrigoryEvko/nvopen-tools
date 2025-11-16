// Function: sub_5C67D0
// Address: 0x5c67d0
//
_QWORD *__fastcall sub_5C67D0(__int64 a1)
{
  __int64 v1; // rax
  char v2; // dl
  __int64 v3; // rcx
  __int64 v4; // r8
  _QWORD *result; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rdx
  __int64 v9; // rcx

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(_BYTE *)(v1 + 80);
  if ( v2 == 7 )
  {
    v8 = *(_QWORD **)(v1 + 88);
    result = (_QWORD *)v8[29];
    if ( result )
    {
      while ( v8 != result )
      {
        if ( unk_4F07588 )
        {
          v9 = result[4];
          if ( v8[4] == v9 )
          {
            if ( v9 )
              break;
          }
        }
        result = (_QWORD *)result[29];
        if ( !result )
          return result;
      }
      v8[29] = 0;
      return (_QWORD *)sub_6851C0(1557, a1 + 32);
    }
  }
  else
  {
    if ( v2 != 11 )
      sub_721090(a1);
    v3 = *(_QWORD *)(v1 + 88);
    v4 = *(_QWORD *)(v3 + 256);
    for ( result = *(_QWORD **)(v4 + 8); result; result = *(_QWORD **)(v7 + 8) )
    {
      v7 = result[32];
      if ( !v7 )
        break;
      if ( (_QWORD *)v3 != result )
      {
        if ( !unk_4F07588 )
          continue;
        v6 = result[4];
        if ( *(_QWORD *)(v3 + 32) != v6 || !v6 )
          continue;
      }
      *(_QWORD *)(v4 + 8) = 0;
      *(_BYTE *)(v3 + 202) &= ~4u;
      return (_QWORD *)sub_6851C0(1557, a1 + 32);
    }
  }
  return result;
}
