// Function: sub_14399D0
// Address: 0x14399d0
//
__int64 __fastcall sub_14399D0(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r12
  _QWORD *v3; // r13
  __int64 v4; // rbx
  unsigned __int8 v5; // al
  __int64 result; // rax
  __int64 v7; // rbx
  _BYTE *v8; // rbx
  unsigned __int8 v9; // al
  __int64 v10; // r12
  unsigned int v11; // ecx
  unsigned int v12; // ecx
  unsigned int v13; // ecx

  if ( *(_BYTE *)(a1 + 16) <= 0x17u )
    return 24;
  switch ( *(_BYTE *)(a1 + 16) )
  {
    case 0x19:
    case 0x1A:
    case 0x1B:
    case 0x1C:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x35:
    case 0x38:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x46:
    case 0x47:
    case 0x4C:
    case 0x4D:
    case 0x4F:
    case 0x52:
    case 0x53:
    case 0x54:
    case 0x55:
    case 0x56:
      goto LABEL_13;
    case 0x1D:
      return sub_1438880(a1 & 0xFFFFFFFFFFFFFFFBLL);
    case 0x4B:
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v7 = *(_QWORD *)(a1 - 8);
      else
        v7 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v8 = *(_BYTE **)(v7 + 24);
      v9 = v8[16];
      if ( v9 <= 0x10u
        || v9 == 53
        || v9 == 17
        && ((unsigned __int8)sub_15E0450(v8)
         || (unsigned __int8)sub_15E0470(v8)
         || (unsigned __int8)sub_15E0490(v8)
         || (unsigned __int8)sub_15E04F0(v8)) )
      {
        goto LABEL_13;
      }
      result = 23;
      if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) != 15 )
        goto LABEL_13;
      return result;
    case 0x4E:
      v10 = *(_QWORD *)(a1 - 24);
      if ( *(_BYTE *)(v10 + 16) )
        return sub_1438880(a1 | 4);
      result = sub_1438F00(*(_QWORD *)(a1 - 24));
      if ( (_DWORD)result != 21 )
        return result;
      v11 = *(_DWORD *)(v10 + 36);
      if ( v11 > 0x94 )
      {
        v12 = v11 - 186;
        if ( v12 > 0x1C || ((1LL << v12) & 0x1C01C001) == 0 )
          return sub_1438880(a1 | 4);
LABEL_13:
        result = 24;
      }
      else
      {
        if ( v11 <= 0x64 )
        {
          if ( v11 - 1 <= 0x33 )
          {
            switch ( v11 )
            {
              case 1u:
              case 2u:
              case 0x24u:
              case 0x25u:
              case 0x26u:
              case 0x29u:
              case 0x2Cu:
              case 0x2Du:
              case 0x2Fu:
              case 0x31u:
              case 0x34u:
                goto LABEL_13;
              default:
                goto LABEL_42;
            }
          }
          return sub_1438880(a1 | 4);
        }
        switch ( v11 )
        {
          case 0x65u:
          case 0x6Du:
          case 0x71u:
          case 0x72u:
          case 0x74u:
          case 0x75u:
          case 0x90u:
          case 0x94u:
            goto LABEL_13;
          default:
LABEL_42:
            v13 = v11 - 133;
            if ( v13 > 4 )
              return sub_1438880(a1 | 4);
            result = 23;
            if ( ((1LL << v13) & 0x15) == 0 )
              return sub_1438880(a1 | 4);
            break;
        }
      }
      break;
    default:
      v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
      v2 = (_QWORD *)(a1 - 24 * v1);
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v2 = *(_QWORD **)(a1 - 8);
      v3 = &v2[3 * v1];
      if ( v3 == v2 )
        goto LABEL_13;
      while ( 1 )
      {
        v4 = *v2;
        v5 = *(_BYTE *)(*v2 + 16LL);
        if ( v5 > 0x10u
          && v5 != 53
          && (v5 != 17
           || !(unsigned __int8)sub_15E0450(*v2)
           && !(unsigned __int8)sub_15E0470(v4)
           && !(unsigned __int8)sub_15E0490(v4)
           && !(unsigned __int8)sub_15E04F0(v4))
          && *(_BYTE *)(*(_QWORD *)v4 + 8LL) == 15 )
        {
          return 23;
        }
        v2 += 3;
        if ( v2 == v3 )
          goto LABEL_13;
      }
  }
  return result;
}
