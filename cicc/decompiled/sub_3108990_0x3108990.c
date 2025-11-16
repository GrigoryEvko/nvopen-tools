// Function: sub_3108990
// Address: 0x3108990
//
__int64 __fastcall sub_3108990(__int64 a1)
{
  __int64 *v1; // r12
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 *v5; // rax
  __int64 *v6; // r13
  _BYTE *v7; // rbx
  unsigned __int8 v8; // al
  __int64 v9; // r12
  __int64 v10; // rbx
  char v11; // al
  unsigned int v12; // eax
  __int64 v13; // rdx

  if ( *(_BYTE *)a1 <= 0x1Cu )
    return 24;
  switch ( *(_BYTE *)a1 )
  {
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x21:
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
    case 0x36:
    case 0x37:
    case 0x38:
    case 0x39:
    case 0x3A:
    case 0x3B:
    case 0x3C:
    case 0x3F:
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x47:
    case 0x48:
    case 0x49:
    case 0x4A:
    case 0x4B:
    case 0x4D:
    case 0x4E:
    case 0x53:
    case 0x54:
    case 0x56:
    case 0x59:
    case 0x5A:
    case 0x5B:
    case 0x5C:
    case 0x5D:
      return 24;
    case 0x22:
      return sub_31087C0((unsigned __int8 *)a1);
    case 0x52:
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        v9 = *(_QWORD *)(a1 - 8);
      else
        v9 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      v10 = *(_QWORD *)(v9 + 32);
      v11 = *(_BYTE *)v10;
      if ( *(_BYTE *)v10 <= 0x15u
        || v11 == 60
        || v11 == 22
        && ((unsigned __int8)sub_B2BAE0(*(_QWORD *)(v9 + 32))
         || (unsigned __int8)sub_B2D6E0(v10)
         || (unsigned __int8)sub_B2D720(v10))
        || *(_BYTE *)(*(_QWORD *)(v10 + 8) + 8LL) != 14 )
      {
        return 24;
      }
      return 23;
    case 0x55:
      v3 = *(_QWORD *)(a1 - 32);
      if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(a1 + 80) != *(_QWORD *)(v3 + 24) )
        return sub_31087C0((unsigned __int8 *)a1);
      result = sub_3108960(*(_QWORD *)(a1 - 32));
      if ( (_DWORD)result != 21 )
        return result;
      v12 = *(_DWORD *)(v3 + 36);
      if ( v12 > 0xD3 )
      {
        if ( v12 > 0x133 )
        {
          if ( v12 > 0x157 )
          {
            if ( v12 - 373 > 2 )
              return sub_31087C0((unsigned __int8 *)a1);
          }
          else if ( v12 <= 0x154 )
          {
            return sub_31087C0((unsigned __int8 *)a1);
          }
        }
        else
        {
          if ( v12 <= 0x119 )
          {
LABEL_38:
            if ( v12 - 238 <= 5 && ((1LL << ((unsigned __int8)v12 + 18)) & 0x29) != 0 )
              return 23;
            return sub_31087C0((unsigned __int8 *)a1);
          }
          if ( ((1LL << ((unsigned __int8)v12 - 26)) & 0x2000011) == 0 )
            return sub_31087C0((unsigned __int8 *)a1);
        }
      }
      else
      {
        if ( v12 <= 0xB1 )
        {
          if ( v12 > 4 )
          {
            switch ( v12 )
            {
              case 'E':
              case 'F':
              case 'G':
              case 'J':
              case 'N':
              case 'O':
              case 'Q':
              case 'S':
              case 'V':
                return 24;
              case 'H':
              case 'I':
              case 'K':
              case 'L':
              case 'M':
              case 'P':
              case 'R':
              case 'T':
              case 'U':
                goto LABEL_38;
              default:
                return sub_31087C0((unsigned __int8 *)a1);
            }
          }
          if ( v12 > 2 )
            return 24;
          return sub_31087C0((unsigned __int8 *)a1);
        }
        v13 = 0x30C020001LL;
        if ( !_bittest64(&v13, v12 - 178) )
          return sub_31087C0((unsigned __int8 *)a1);
      }
      return 24;
    default:
      v4 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      v5 = (__int64 *)(a1 - v4 * 8);
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        v5 = *(__int64 **)(a1 - 8);
      v6 = &v5[v4];
      v1 = v5;
      if ( v6 == v5 )
        return 24;
      while ( 1 )
      {
        v7 = (_BYTE *)*v1;
        v8 = *(_BYTE *)*v1;
        if ( v8 > 0x15u
          && v8 != 60
          && (v8 != 22
           || !(unsigned __int8)sub_B2BAE0(*v1)
           && !(unsigned __int8)sub_B2D6E0((__int64)v7)
           && !(unsigned __int8)sub_B2D720((__int64)v7))
          && *(_BYTE *)(*((_QWORD *)v7 + 1) + 8LL) == 14 )
        {
          break;
        }
        v1 += 4;
        if ( v6 == v1 )
          return 24;
      }
      return 23;
  }
}
