// Function: sub_9860E0
// Address: 0x9860e0
//
char __fastcall sub_9860E0(unsigned __int8 *a1, unsigned __int64 a2, char a3)
{
  unsigned __int8 v3; // cl
  int v4; // ebx
  unsigned __int64 v5; // rax
  char v6; // r8
  __int64 v7; // rdx
  __int64 *v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned int v11; // r12d
  unsigned __int64 v12; // r13
  _DWORD *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rcx
  _DWORD *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rcx
  _DWORD *v19; // rax
  char *v20; // r12
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  _QWORD v25[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a3 && (a2 & 1) != 0 && (unsigned __int8)sub_BB5510() )
    goto LABEL_12;
  v3 = *a1;
  if ( *a1 <= 0x1Cu )
  {
    v4 = *((unsigned __int16 *)a1 + 1);
    switch ( *((_WORD *)a1 + 1) )
    {
      case 5:
      case 0xB:
      case 0x38:
LABEL_17:
        v6 = sub_A74710(a1 + 72, 0, 40);
        LOBYTE(v5) = 0;
        if ( !v6 )
        {
          v7 = *((_QWORD *)a1 - 4);
          LOBYTE(v5) = 1;
          if ( v7 )
          {
            if ( !*(_BYTE *)v7 && *(_QWORD *)(v7 + 24) == *((_QWORD *)a1 + 10) )
            {
              v25[0] = *(_QWORD *)(v7 + 120);
              LODWORD(v5) = sub_A74710(v25, 0, 40) ^ 1;
            }
          }
        }
        return v5;
      case 0xC:
      case 0xE:
      case 0x10:
      case 0x12:
      case 0x15:
      case 0x16:
      case 0x17:
      case 0x18:
      case 0x22:
      case 0x35:
      case 0x36:
      case 0x37:
      case 0x39:
      case 0x40:
      case 0x41:
      case 0x43:
        goto LABEL_6;
      case 0x19:
      case 0x1A:
      case 0x1B:
LABEL_5:
        if ( (a2 & 1) == 0 )
          goto LABEL_6;
        if ( (a1[7] & 0x40) != 0 )
          v20 = (char *)*((_QWORD *)a1 - 1);
        else
          v20 = (char *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        LODWORD(v5) = sub_985C10(*((_QWORD *)v20 + 4), a2) ^ 1;
        return v5;
      case 0x29:
      case 0x2A:
        goto LABEL_12;
      case 0x3D:
      case 0x3E:
LABEL_22:
        if ( (a1[7] & 0x40) != 0 )
          v8 = (__int64 *)*((_QWORD *)a1 - 1);
        else
          v8 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        v5 = 64;
        if ( v4 != 62 )
          v5 = 32;
        v9 = *(__int64 *)((char *)v8 + v5);
        LOBYTE(v5) = a2 & 1;
        if ( *(_BYTE *)v9 != 17 )
          return v5;
        if ( !(_BYTE)v5 )
          goto LABEL_6;
        v10 = *v8;
        v11 = *(_DWORD *)(v9 + 32);
        v12 = *(unsigned int *)(*(_QWORD *)(v10 + 8) + 32LL);
        if ( v11 > 0x40 )
        {
          if ( v11 - (unsigned int)sub_C444A0(v9 + 24) > 0x40 )
          {
LABEL_12:
            LOBYTE(v5) = 1;
            return v5;
          }
          v5 = **(_QWORD **)(v9 + 24);
        }
        else
        {
          v5 = *(_QWORD *)(v9 + 24);
        }
        LOBYTE(v5) = v12 <= v5;
        return v5;
      case 0x3F:
LABEL_31:
        if ( v3 == 5 )
        {
          v22 = sub_AC35F0(a1);
          v14 = v23;
          v13 = (_DWORD *)v22;
        }
        else
        {
          v13 = (_DWORD *)*((_QWORD *)a1 + 9);
          v14 = *((unsigned int *)a1 + 20);
        }
        LOBYTE(v5) = a2 & 1;
        if ( (a2 & 1) == 0 )
          return v5;
        v15 = 4 * v14;
        v16 = &v13[(unsigned __int64)v15 / 4];
        v17 = v15 >> 4;
        v18 = v15 >> 2;
        if ( v17 <= 0 )
          goto LABEL_58;
        v19 = &v13[4 * v17];
        do
        {
          if ( *v13 == -1 )
            goto LABEL_41;
          if ( v13[1] == -1 )
          {
            LOBYTE(v5) = v16 != v13 + 1;
            return v5;
          }
          if ( v13[2] == -1 )
          {
            LOBYTE(v5) = v16 != v13 + 2;
            return v5;
          }
          if ( v13[3] == -1 )
          {
            LOBYTE(v5) = v16 != v13 + 3;
            return v5;
          }
          v13 += 4;
        }
        while ( v19 != v13 );
        v18 = v16 - v13;
LABEL_58:
        if ( v18 == 2 )
          goto LABEL_68;
        if ( v18 == 3 )
        {
          LOBYTE(v5) = v13 != v16;
          if ( *v13 == -1 )
            return v5;
          ++v13;
LABEL_68:
          if ( *v13 != -1 )
          {
            ++v13;
            goto LABEL_61;
          }
LABEL_41:
          LOBYTE(v5) = v16 != v13;
          return v5;
        }
        if ( v18 != 1 )
        {
LABEL_6:
          LOBYTE(v5) = 0;
          return v5;
        }
LABEL_61:
        LOBYTE(v5) = 0;
        if ( *v13 == -1 )
          goto LABEL_41;
        return v5;
      default:
        if ( v3 == 5 && (unsigned __int8)sub_AC35E0(a1) )
          goto LABEL_6;
LABEL_10:
        LOBYTE(v5) = (unsigned int)(v4 - 13) > 0x11;
        return v5;
    }
  }
  v4 = v3 - 29;
  switch ( v3 )
  {
    case '"':
    case '(':
      goto LABEL_17;
    case ')':
    case '+':
    case '-':
    case '/':
    case '2':
    case '3':
    case '4':
    case '5':
    case '?':
    case 'R':
    case 'S':
    case 'T':
    case 'V':
    case ']':
    case '^':
    case '`':
      goto LABEL_6;
    case '6':
    case '7':
    case '8':
      goto LABEL_5;
    case 'F':
    case 'G':
      goto LABEL_12;
    case 'U':
      v21 = *((_QWORD *)a1 - 4);
      if ( !v21 || *(_BYTE *)v21 || *(_QWORD *)(v21 + 24) != *((_QWORD *)a1 + 10) || (*(_BYTE *)(v21 + 33) & 0x20) == 0 )
        goto LABEL_17;
      switch ( *(_DWORD *)(v21 + 36) )
      {
        case 1:
        case 0x41:
        case 0x43:
          if ( !(unsigned __int8)sub_AC30F0(*(_QWORD *)&a1[32 * (1LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))]) )
            goto LABEL_17;
          goto LABEL_6;
        case 8:
        case 0xE:
        case 0xF:
        case 0x14:
        case 0x15:
        case 0x1A:
        case 0x3F:
        case 0x42:
        case 0x58:
        case 0x59:
        case 0x5A:
        case 0xAA:
        case 0xAC:
        case 0xAD:
        case 0xAE:
        case 0xAF:
        case 0xB0:
        case 0xB1:
        case 0xB3:
        case 0xB4:
        case 0xB5:
        case 0xCF:
        case 0xD1:
        case 0xD4:
        case 0xD5:
        case 0xDA:
        case 0xDB:
        case 0xDC:
        case 0xDF:
        case 0xE0:
        case 0xEB:
        case 0xED:
        case 0xF6:
        case 0xF8:
        case 0xFA:
        case 0x11C:
        case 0x11D:
        case 0x12B:
        case 0x134:
        case 0x135:
        case 0x136:
        case 0x137:
        case 0x138:
        case 0x145:
        case 0x149:
        case 0x14A:
        case 0x14D:
        case 0x14F:
        case 0x152:
        case 0x153:
        case 0x163:
        case 0x167:
        case 0x168:
        case 0x16D:
        case 0x16E:
        case 0x171:
        case 0x173:
        case 0x174:
          goto LABEL_6;
        case 0x151:
        case 0x172:
          if ( (a2 & 1) == 0 )
            goto LABEL_6;
          LODWORD(v5) = sub_985C10(*(_QWORD *)&a1[32 * (1LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))], a2) ^ 1;
          break;
        default:
          goto LABEL_17;
      }
      break;
    case 'Z':
    case '[':
      goto LABEL_22;
    case '\\':
      goto LABEL_31;
    default:
      if ( (unsigned int)v3 - 67 > 0xC )
        goto LABEL_10;
      goto LABEL_6;
  }
  return v5;
}
