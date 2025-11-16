// Function: sub_1847CC0
// Address: 0x1847cc0
//
__int64 __fastcall sub_1847CC0(__int64 a1)
{
  unsigned int v1; // r8d
  __int64 v2; // r13
  __int64 v3; // rbx
  char v4; // al
  __int64 v5; // rbx
  __int64 i; // r12
  __int64 v8; // r15
  char v9; // al

  v1 = 0;
  v2 = a1 + 8;
  v3 = *(_QWORD *)(a1 + 16);
  if ( v3 != a1 + 8 )
  {
    do
    {
      while ( 1 )
      {
        if ( !v3 )
          BUG();
        if ( (*(_BYTE *)(v3 - 24) & 0xF) == 1 )
        {
          if ( !sub_15E4F60(v3 - 56) )
          {
            v8 = *(_QWORD *)(v3 - 80);
            sub_15E5440(v3 - 56, 0);
            if ( (unsigned __int8)sub_1ACF050(v8) )
              sub_159D850(v8);
          }
          sub_159D9E0(v3 - 56);
          v4 = *(_BYTE *)(v3 - 24);
          v1 = 1;
          *(_BYTE *)(v3 - 24) = v4 & 0xF0;
          if ( (v4 & 0x30) != 0 )
            break;
        }
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          goto LABEL_9;
      }
      *(_BYTE *)(v3 - 23) |= 0x40u;
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v2 != v3 );
  }
LABEL_9:
  v5 = *(_QWORD *)(a1 + 32);
  for ( i = a1 + 24; i != v5; v1 = 1 )
  {
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      if ( (*(_BYTE *)(v5 - 24) & 0xF) == 1 )
        break;
      v5 = *(_QWORD *)(v5 + 8);
      if ( i == v5 )
        return v1;
    }
    if ( !sub_15E4F60(v5 - 56) )
    {
      sub_15E0C30(v5 - 56);
      v9 = *(_BYTE *)(v5 - 24);
      *(_BYTE *)(v5 - 24) = v9 & 0xF0;
      if ( (v9 & 0x30) != 0 )
        *(_BYTE *)(v5 - 23) |= 0x40u;
    }
    sub_159D9E0(v5 - 56);
    v5 = *(_QWORD *)(v5 + 8);
  }
  return v1;
}
