// Function: sub_8782F0
// Address: 0x8782f0
//
_BOOL8 __fastcall sub_8782F0(__int64 a1)
{
  _BOOL8 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  char v5; // cl

  switch ( *(_BYTE *)(a1 + 80) )
  {
    case 2:
      v2 = *(_QWORD *)(a1 + 88);
      result = 0;
      if ( *(_BYTE *)(v2 + 173) == 12 )
        return (*(_BYTE *)(v2 + 177) & 4) != 0;
      return result;
    case 3:
      v4 = *(_QWORD *)(a1 + 88);
      v5 = *(_BYTE *)(v4 + 140);
      if ( v5 != 12 )
        goto LABEL_13;
      result = 0;
      if ( *(_BYTE *)(v4 + 184) != 10 )
      {
        do
        {
          v4 = *(_QWORD *)(v4 + 160);
          v5 = *(_BYTE *)(v4 + 140);
        }
        while ( v5 == 12 );
LABEL_13:
        result = 0;
        if ( v5 == 14 )
          return *(_BYTE *)(v4 + 161) & 1;
      }
      return result;
    case 7:
      return (*(_BYTE *)(*(_QWORD *)(a1 + 88) + 175LL) & 0x20) != 0;
    case 8:
      return (*(_WORD *)(*(_QWORD *)(a1 + 88) + 144LL) & 0x280) == 640;
    case 0x12:
      return *(_BYTE *)(*(_QWORD *)(a1 + 88) + 42LL) & 1;
    case 0x13:
      v3 = *(_QWORD *)(a1 + 88);
      if ( (*(_BYTE *)(v3 + 266) & 1) != 0 )
        v3 = *(_QWORD *)(*(_QWORD *)(v3 + 200) + 88LL);
      return (*(_BYTE *)(*(_QWORD *)(v3 + 104) + 121LL) & 4) != 0;
    default:
      return 0;
  }
}
