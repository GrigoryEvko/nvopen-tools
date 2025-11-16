// Function: sub_890340
// Address: 0x890340
//
_QWORD *__fastcall sub_890340(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 v3; // r14
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  _QWORD *result; // rax

  v1 = *(_QWORD *)(a1 + 24);
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 24);
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 24);
      if ( v3 )
      {
        if ( *(_QWORD *)(v3 + 24) )
          sub_890340();
        v4 = *(_QWORD **)v3;
        if ( *(_QWORD *)v3 )
        {
          do
          {
            *(_BYTE *)(v4[1] + 83LL) |= 0x40u;
            v4 = (_QWORD *)*v4;
          }
          while ( v4 );
        }
      }
      v5 = *(_QWORD **)v2;
      if ( *(_QWORD *)v2 )
      {
        do
        {
          *(_BYTE *)(v5[1] + 83LL) |= 0x40u;
          v5 = (_QWORD *)*v5;
        }
        while ( v5 );
      }
    }
    v6 = *(_QWORD **)v1;
    if ( *(_QWORD *)v1 )
    {
      do
      {
        *(_BYTE *)(v6[1] + 83LL) |= 0x40u;
        v6 = (_QWORD *)*v6;
      }
      while ( v6 );
    }
  }
  result = *(_QWORD **)a1;
  if ( *(_QWORD *)a1 )
  {
    do
    {
      *(_BYTE *)(result[1] + 83LL) |= 0x40u;
      result = (_QWORD *)*result;
    }
    while ( result );
  }
  return result;
}
