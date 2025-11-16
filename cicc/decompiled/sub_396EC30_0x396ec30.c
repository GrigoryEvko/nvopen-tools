// Function: sub_396EC30
// Address: 0x396ec30
//
__int64 __fastcall sub_396EC30(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // r9
  __int64 v5; // rdx
  unsigned __int16 v6; // ax
  __int64 v7; // kr00_8
  __int64 v8; // rsi
  __int64 v9; // rdi

  result = *(_DWORD *)(a1[30] + 348LL) & 0xFFFFFFFD;
  if ( (_DWORD)result == 1 )
  {
    result = sub_396EB00((__int64)a1);
    if ( (_DWORD)result )
    {
      v4 = a2[3];
      v5 = a2[1];
      if ( v4 + 24 != v5 )
      {
        while ( 2 )
        {
          v6 = **(_WORD **)(v5 + 16);
          v7 = v3;
          v3 = v6;
          switch ( v6 )
          {
            case 0u:
            case 8u:
            case 0xAu:
            case 0xEu:
            case 0xFu:
            case 0x2Du:
LABEL_6:
              v5 = *(_QWORD *)(v5 + 8);
              if ( v4 + 24 == v5 )
                break;
              continue;
            default:
              v3 = v7;
              switch ( v6 )
              {
                case 2u:
                case 3u:
                case 4u:
                case 6u:
                case 9u:
                case 0xCu:
                case 0xDu:
                case 0x11u:
                case 0x12u:
                  goto LABEL_6;
                default:
                  goto LABEL_8;
              }
          }
          break;
        }
      }
      result = *(_QWORD *)(*(_QWORD *)(v4 + 56) + 320LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v4 != result )
      {
LABEL_8:
        v8 = *(_QWORD *)(a1[33] + 384LL) + 48LL * *(unsigned int *)(a2[4] + 24LL);
        v9 = a1[32];
        switch ( *(_DWORD *)v8 )
        {
          case 0:
            result = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v9 + 784LL))(
                       v9,
                       *(unsigned int *)(v8 + 16));
            break;
          case 1:
          case 2:
          case 7:
          case 0xB:
          case 0xE:
            result = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v9 + 832LL))(v9, *(int *)(v8 + 20));
            break;
          case 3:
            result = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v9 + 744LL))(
                       v9,
                       *(unsigned int *)(v8 + 16),
                       *(int *)(v8 + 20));
            break;
          case 4:
            result = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v9 + 736LL))(
                       v9,
                       *(unsigned int *)(v8 + 16));
            break;
          case 5:
            result = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v9 + 728LL))(v9, *(int *)(v8 + 20));
            break;
          case 6:
            result = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v9 + 720LL))(
                       v9,
                       *(unsigned int *)(v8 + 16),
                       *(int *)(v8 + 20));
            break;
          case 8:
            result = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v9 + 808LL))(v9, *(int *)(v8 + 20));
            break;
          case 9:
            result = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v9 + 816LL))(
                       v9,
                       *(_QWORD *)(v8 + 24),
                       *(_QWORD *)(v8 + 32) - *(_QWORD *)(v8 + 24));
            break;
          case 0xA:
            result = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v9 + 792LL))(
                       v9,
                       *(unsigned int *)(v8 + 16));
            break;
          case 0xC:
            result = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v9 + 856LL))(
                       v9,
                       *(unsigned int *)(v8 + 16),
                       *(unsigned int *)(v8 + 20));
            break;
          case 0xD:
            result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 864LL))(v9);
            break;
        }
      }
    }
  }
  return result;
}
