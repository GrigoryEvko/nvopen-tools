// Function: sub_7F5750
// Address: 0x7f5750
//
__int64 __fastcall sub_7F5750(__int64 a1, _QWORD *a2)
{
  __int64 i; // rax
  __int64 v4; // r12
  __int64 *v5; // rax
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rcx
  __int64 v9; // rdi
  __int64 v10; // r8
  __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  _QWORD *v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rsi

  while ( 1 )
  {
    for ( i = *(_QWORD *)(a1 + 128); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v4 = *(_QWORD *)(a1 + 176);
    if ( !v4 )
      break;
    while ( (*(_BYTE *)(v4 + 171) & 0x20) != 0 )
    {
      v4 = *(_QWORD *)(v4 + 120);
      if ( !v4 )
        goto LABEL_32;
    }
    v5 = (__int64 *)sub_72FD90(*(_QWORD *)(i + 160), 11);
    if ( !v5 )
      return v4;
    while ( 1 )
    {
      v6 = a2[13];
      v7 = v5[16];
      if ( (a2[12] & 1) != 0 )
      {
        if ( v6 != v7 )
          goto LABEL_12;
        return v4;
      }
      if ( v6 >= v7 )
      {
        v9 = v5[15];
        v10 = *(unsigned __int8 *)(v9 + 140);
        v11 = v9;
        if ( (_BYTE)v10 == 12 )
        {
          do
            v11 = *(_QWORD *)(v11 + 160);
          while ( *(_BYTE *)(v11 + 140) == 12 );
        }
        v12 = *(_QWORD *)(v11 + 128) + v7;
        if ( v12 > v6 )
          break;
      }
      do
      {
LABEL_12:
        v4 = *(_QWORD *)(v4 + 120);
        if ( !v4 )
        {
          v4 = 0;
          sub_72FD90(v5[14], 11);
          return v4;
        }
      }
      while ( (*(_BYTE *)(v4 + 171) & 0x20) != 0 );
      v5 = (__int64 *)sub_72FD90(v5[14], 11);
      if ( !v5 )
        return v4;
    }
    if ( (_BYTE)v10 == 12 )
    {
      do
        v9 = *(_QWORD *)(v9 + 160);
      while ( *(_BYTE *)(v9 + 140) == 12 );
    }
    v13 = *(_QWORD *)(v9 + 168);
    if ( (*(_BYTE *)(v13 + 109) & 0x10) == 0 )
      v13 = *(_QWORD *)(*(_QWORD *)(v13 + 208) + 168LL);
    v14 = *(_QWORD **)v13;
    if ( *(_QWORD *)v13 )
    {
      while ( 1 )
      {
        v15 = v14[5];
        v16 = a2[5];
        if ( v15 == v16 || (unsigned int)sub_8D97D0(v15, v16, 0, v12, v10) )
        {
          if ( (unsigned int)sub_5ED650(
                               *(_QWORD **)(v14[14] + 8LL),
                               *(_QWORD **)(v14[14] + 16LL),
                               **(_QWORD ***)(a2[14] + 8LL),
                               *(_QWORD **)(a2[14] + 16LL)) )
            break;
        }
        v14 = (_QWORD *)*v14;
        if ( !v14 )
          goto LABEL_30;
      }
      a2 = v14;
    }
    else
    {
LABEL_30:
      a2 = 0;
    }
    *(_BYTE *)(v4 + 171) |= 0x10u;
    a1 = v4;
  }
LABEL_32:
  v4 = 0;
  sub_72FD90(*(_QWORD *)(i + 160), 11);
  return v4;
}
