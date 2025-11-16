// Function: sub_1C6D5B0
// Address: 0x1c6d5b0
//
__int64 __fastcall sub_1C6D5B0(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 v2; // r12
  __int64 v3; // r15
  __int64 v4; // rbx
  char v5; // al

  v1 = 0;
  v2 = a1 + 24;
  v3 = *(_QWORD *)(a1 + 32);
  if ( v3 != a1 + 24 )
  {
    while ( 1 )
    {
      v4 = v3 - 56;
      if ( !v3 )
        v4 = 0;
      if ( sub_15E4F60(v4) )
        goto LABEL_3;
      v1 = sub_1C2F070(v4);
      v5 = *(_BYTE *)(v4 + 32);
      if ( (_BYTE)v1 )
      {
        *(_BYTE *)(v4 + 32) = v5 & 0xF0;
        if ( (v5 & 0x30) != 0 )
          goto LABEL_9;
LABEL_3:
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          return v1;
      }
      else
      {
        *(_BYTE *)(v4 + 32) = v5 & 0xC0 | 7;
LABEL_9:
        *(_BYTE *)(v4 + 33) |= 0x40u;
        v3 = *(_QWORD *)(v3 + 8);
        v1 = 1;
        if ( v2 == v3 )
          return v1;
      }
    }
  }
  return v1;
}
