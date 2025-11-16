// Function: sub_733F40
// Address: 0x733f40
//
__int64 sub_733F40()
{
  __int64 v0; // r12
  __int64 v1; // rcx
  __int64 v2; // rax
  _QWORD *v3; // rsi
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  bool v7; // zf

  v0 = qword_4F06BC0;
  qword_4F06BC0 = *(_QWORD *)(qword_4F06BC0 + 32LL);
  if ( *(_BYTE *)v0 == 1 && *(_BYTE *)(v0 + 8) == 23 && *(_BYTE *)(*(_QWORD *)(v0 + 16) + 28LL) == 17 )
  {
    if ( !(unsigned int)sub_733920(v0) )
    {
      *(_BYTE *)(*(_QWORD *)(v0 + 32) + 1LL) |= 2u;
      return 1;
    }
  }
  else
  {
    if ( !(unsigned int)sub_733920(v0) )
      return 1;
    v1 = *(_QWORD *)(v0 + 32);
    if ( v1 )
    {
      v2 = *(_QWORD *)(v1 + 48);
      v3 = (_QWORD *)(v1 + 48);
      if ( v2 != v0 )
      {
        do
        {
          v4 = v2;
          v2 = *(_QWORD *)(v2 + 56);
        }
        while ( v2 != v0 );
        v3 = (_QWORD *)(v4 + 56);
      }
      v5 = *(_QWORD **)(v0 + 48);
      if ( !v5 )
        goto LABEL_22;
      do
      {
        v5[4] = v1;
        v5[5] = *(_QWORD *)(v0 + 40);
        v6 = v5;
        v5 = (_QWORD *)v5[7];
      }
      while ( v5 );
      if ( *(_QWORD *)(v0 + 48) )
      {
        v6[7] = *(_QWORD *)(v0 + 56);
        *v3 = *(_QWORD *)(v0 + 48);
      }
      else
      {
LABEL_22:
        *v3 = *(_QWORD *)(v0 + 56);
      }
      if ( *(_BYTE *)v0 == 2 && (*(_BYTE *)(v0 + 1) & 1) == 0 )
        *(_BYTE *)(v1 + 1) &= ~1u;
    }
  }
  *(_QWORD *)(v0 + 32) = 0;
  v7 = *(_QWORD *)(v0 + 16) == 0;
  *(_QWORD *)(v0 + 48) = 0;
  *(_QWORD *)(v0 + 56) = 0;
  if ( !v7 )
    sub_733650(v0);
  sub_732E20(v0);
  return 0;
}
