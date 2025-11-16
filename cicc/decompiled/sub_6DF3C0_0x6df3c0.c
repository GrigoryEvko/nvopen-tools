// Function: sub_6DF3C0
// Address: 0x6df3c0
//
__int64 __fastcall sub_6DF3C0(__int64 a1, __int64 *a2)
{
  unsigned int v2; // r8d
  char v3; // al
  __int64 v4; // rax

  if ( dword_4F077C4 == 1 )
    return 0;
  if ( unk_4D048BC )
  {
    v2 = dword_4F077BC;
    while ( 1 )
    {
      v3 = *(_BYTE *)(a1 + 24);
      if ( v3 != 1 )
        break;
LABEL_8:
      switch ( *(_BYTE *)(a1 + 56) )
      {
        case 0x19:
        case 0x49:
        case 0x4A:
        case 0x4B:
        case 0x4C:
        case 0x4D:
        case 0x4E:
        case 0x4F:
        case 0x50:
        case 0x51:
        case 0x52:
        case 0x53:
        case 0x54:
        case 0x55:
          goto LABEL_13;
        case 0x23:
        case 0x24:
          if ( dword_4F077C4 == 2 && !dword_4F077BC )
            return v2;
          goto LABEL_10;
        case 0x25:
        case 0x26:
LABEL_10:
          if ( (_DWORD)qword_4F077B4 )
            return 0;
LABEL_13:
          a1 = *(_QWORD *)(a1 + 72);
          break;
        case 0x5B:
          a1 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL);
          continue;
        case 0x5E:
        case 0x5F:
          v2 = 0;
          if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL) + 56LL) + 144LL) & 4) != 0 )
            goto LABEL_15;
          return v2;
        default:
          return 0;
      }
    }
    while ( v3 == 3 )
    {
      v4 = *(_QWORD *)(a1 + 56);
      if ( *(_BYTE *)(v4 + 177) != 5 )
        break;
      a1 = *(_QWORD *)(v4 + 184);
      v3 = *(_BYTE *)(a1 + 24);
      if ( v3 == 1 )
        goto LABEL_8;
    }
    return 0;
  }
  v2 = sub_6DEAC0(a1);
  if ( !v2 )
    return v2;
  if ( *(_BYTE *)(a1 + 24) == 3 )
    a1 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 184LL);
LABEL_15:
  *a2 = sub_6DF050((__int64 *)a1);
  return 1;
}
