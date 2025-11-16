// Function: sub_216F240
// Address: 0x216f240
//
__int64 __fastcall sub_216F240(_DWORD *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 i; // r13
  unsigned int v5; // r15d
  __int64 v7; // rax
  __int64 v8; // rdx

  v2 = a2 + 72;
  v3 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 == v3 )
    return 0;
  if ( !v3 )
    BUG();
  while ( 1 )
  {
    i = *(_QWORD *)(v3 + 24);
    if ( i != v3 + 16 )
      break;
    v3 = *(_QWORD *)(v3 + 8);
    if ( v2 == v3 )
      return 0;
    if ( !v3 )
      BUG();
  }
  if ( v3 == v2 )
  {
    return 0;
  }
  else
  {
    v5 = 0;
    do
    {
      if ( !i )
        BUG();
      if ( *(_BYTE *)(i - 8) == 78 )
      {
        v7 = *(_QWORD *)(i - 48);
        if ( !*(_BYTE *)(v7 + 16) )
        {
          v8 = i - 24;
          switch ( *(_DWORD *)(v7 + 36) )
          {
            case 0x10BE:
              v5 |= sub_216F100(0, (unsigned int)a1[42], v8);
              break;
            case 0x10BF:
              v5 |= sub_216F100(0, (unsigned int)a1[43], v8);
              break;
            case 0x10C0:
              v5 |= sub_216F100(0, (unsigned int)a1[44], v8);
              break;
            case 0x10E2:
              v5 |= sub_216F100(0, 32, v8);
              break;
            case 0x10E9:
              v5 |= sub_216F100(1, (unsigned int)(a1[42] + 1), v8);
              break;
            case 0x10EA:
              v5 |= sub_216F100(1, (unsigned int)(a1[43] + 1), v8);
              break;
            case 0x10EB:
              v5 |= sub_216F100(1, (unsigned int)(a1[44] + 1), v8);
              break;
            case 0x10EE:
              v5 |= sub_216F100(1, (unsigned int)(a1[39] + 1), v8);
              break;
            case 0x10EF:
              v5 |= sub_216F100(1, (unsigned int)(a1[40] + 1), v8);
              break;
            case 0x10F0:
              v5 |= sub_216F100(1, (unsigned int)(a1[41] + 1), v8);
              break;
            case 0x10F8:
              v5 |= sub_216F100(0, (unsigned int)a1[39], v8);
              break;
            case 0x10F9:
              v5 |= sub_216F100(0, (unsigned int)a1[40], v8);
              break;
            case 0x10FA:
              v5 |= sub_216F100(0, (unsigned int)a1[41], v8);
              break;
            case 0x10FC:
              v5 |= sub_216F100(32, 33, v8);
              break;
            default:
              break;
          }
        }
      }
      for ( i = *(_QWORD *)(i + 8); i == v3 - 24 + 40; i = *(_QWORD *)(v3 + 24) )
      {
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          return v5;
        if ( !v3 )
          BUG();
      }
    }
    while ( v2 != v3 );
  }
  return v5;
}
