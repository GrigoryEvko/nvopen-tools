// Function: sub_72AA80
// Address: 0x72aa80
//
__int64 __fastcall sub_72AA80(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rdx

  switch ( *(_BYTE *)(a1 + 173) )
  {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 7:
    case 0xD:
    case 0xE:
      goto LABEL_5;
    case 6:
      switch ( *(_BYTE *)(a1 + 176) )
      {
        case 0:
        case 4:
        case 5:
          goto LABEL_5;
        case 1:
        case 2:
        case 3:
          result = (*(_BYTE *)(*(_QWORD *)(a1 + 184) - 8LL) & 1) == 0;
          goto LABEL_4;
        case 6:
          return 1;
        default:
          goto LABEL_23;
      }
    case 8:
      return 1;
    case 9:
      v5 = *(_QWORD *)(a1 + 176);
      switch ( *(_BYTE *)(v5 + 48) )
      {
        case 0:
        case 1:
          goto LABEL_5;
        case 2:
          result = sub_72AA80(*(_QWORD *)(v5 + 56));
          goto LABEL_4;
        case 3:
        case 4:
        case 7:
          v1 = *(_QWORD *)(v5 + 56);
          goto LABEL_3;
        case 5:
          result = (*(_BYTE *)(*(_QWORD *)(v5 + 64) - 8LL) & 1) == 0;
          goto LABEL_4;
        default:
          goto LABEL_23;
      }
    case 0xA:
      v1 = *(_QWORD *)(a1 + 176);
      if ( !v1 )
        goto LABEL_5;
LABEL_3:
      result = (*(_BYTE *)(v1 - 8) & 1) == 0;
LABEL_4:
      if ( !(_DWORD)result )
      {
LABEL_5:
        v3 = *(_QWORD *)(a1 + 144);
        result = 0;
        if ( v3 )
          return (*(_BYTE *)(v3 - 8) & 1) == 0;
      }
      return result;
    case 0xB:
      result = (*(_BYTE *)(*(_QWORD *)(a1 + 176) - 8LL) & 1) == 0;
      goto LABEL_4;
    case 0xC:
      switch ( *(_BYTE *)(a1 + 176) )
      {
        case 0:
        case 2:
        case 3:
        case 0xB:
        case 0xD:
          goto LABEL_5;
        case 1:
          goto LABEL_8;
        case 4:
        case 0xC:
          result = sub_72AA80(*(_QWORD *)(a1 + 184));
          goto LABEL_4;
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 0xA:
          v1 = *(_QWORD *)(a1 + 192);
          if ( !v1 )
            goto LABEL_5;
          goto LABEL_3;
        default:
          goto LABEL_23;
      }
    case 0xF:
LABEL_8:
      v4 = *(_QWORD *)(a1 + 184);
      if ( !v4 )
        goto LABEL_5;
      result = 1;
      if ( (*(_BYTE *)(v4 - 8) & 1) != 0 )
        goto LABEL_5;
      return result;
    default:
LABEL_23:
      sub_721090();
  }
}
