// Function: sub_10FD5E0
// Address: 0x10fd5e0
//
__int64 __fastcall sub_10FD5E0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r12d
  __int64 v4; // rdx
  __int64 v6; // r13
  _QWORD *v7; // rax
  _QWORD *v8; // r13
  _QWORD *v9; // rbx
  _QWORD *v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rbx

  while ( 2 )
  {
    v3 = sub_10FD310((_BYTE *)a1, a2);
    if ( (_BYTE)v3 )
      return 1;
    if ( *(_BYTE *)a1 > 0x1Cu )
    {
      v4 = *(_QWORD *)(a1 + 16);
      if ( v4 )
      {
        if ( !*(_QWORD *)(v4 + 8) )
        {
          switch ( *(_BYTE *)a1 )
          {
            case '*':
            case ',':
            case '.':
            case '9':
            case ':':
            case ';':
              if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
                v10 = *(_QWORD **)(a1 - 8);
              else
                v10 = (_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
              if ( !(unsigned __int8)sub_10FD5E0(*v10, a2) )
                return v3;
              if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
                v11 = *(_QWORD *)(a1 - 8);
              else
                v11 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
              a1 = *(_QWORD *)(v11 + 32);
              continue;
            case 'C':
            case 'D':
            case 'E':
              return 1;
            case 'T':
              v6 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
              v7 = (_QWORD *)(a1 - v6 * 8);
              if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
                v7 = *(_QWORD **)(a1 - 8);
              v8 = &v7[v6];
              if ( v8 == v7 )
                return 1;
              v9 = v7;
              break;
            case 'V':
              if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
                v12 = *(_QWORD *)(a1 - 8);
              else
                v12 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
              if ( !(unsigned __int8)sub_10FD5E0(*(_QWORD *)(v12 + 32), a2) )
                return v3;
              if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
                v13 = *(_QWORD *)(a1 - 8);
              else
                v13 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
              a1 = *(_QWORD *)(v13 + 64);
              continue;
            default:
              return v3;
          }
          while ( (unsigned __int8)sub_10FD5E0(*v9, a2) )
          {
            v9 += 4;
            if ( v8 == v9 )
              return 1;
          }
        }
      }
    }
    break;
  }
  return v3;
}
