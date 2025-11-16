// Function: sub_2CB9400
// Address: 0x2cb9400
//
__int64 __fastcall sub_2CB9400(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 v2; // r12
  __int64 v3; // r15
  __int64 v4; // rbx
  char v6; // al

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
      if ( sub_B2FC80(v4) )
        goto LABEL_3;
      v1 = sub_CE9220(v4);
      if ( (_BYTE)v1 )
      {
        v6 = *(_BYTE *)(v4 + 32);
        *(_BYTE *)(v4 + 32) = v6 & 0xF0;
        if ( (v6 & 0x30) != 0 )
          goto LABEL_9;
LABEL_3:
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          return v1;
      }
      else
      {
        *(_WORD *)(v4 + 32) = *(_WORD *)(v4 + 32) & 0xFCC0 | 7;
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
