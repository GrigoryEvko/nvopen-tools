// Function: sub_17ADFE0
// Address: 0x17adfe0
//
__int64 __fastcall sub_17ADFE0(__int64 a1, _DWORD *a2, __int64 a3, int a4)
{
  unsigned int v4; // r12d
  int v5; // eax
  int v8; // ecx
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v13; // r15
  _QWORD *v14; // r8
  unsigned int v15; // r13d
  _QWORD *v16; // r15
  _QWORD *v17; // r9
  __int64 v18; // r8
  int v19; // esi
  _DWORD *v20; // rax
  char v21; // di
  int v22; // eax
  __int64 v23; // [rsp+0h] [rbp-50h]
  _QWORD *v24; // [rsp+8h] [rbp-48h]
  int v25; // [rsp+14h] [rbp-3Ch]
  _QWORD *v26; // [rsp+18h] [rbp-38h]
  unsigned int v27; // [rsp+18h] [rbp-38h]

  v5 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned __int8)v5 <= 0x10u )
  {
    return 1;
  }
  else
  {
    v8 = a3;
    v10 = (__int64)&a2[(unsigned int)(a3 - 1) + 1];
LABEL_3:
    if ( (unsigned __int8)v5 > 0x17u )
    {
      v11 = *(_QWORD *)(a1 + 8);
      if ( v11 )
      {
        LOBYTE(v4) = a4 == 0 || *(_QWORD *)(v11 + 8) != 0;
        if ( !(_BYTE)v4 )
        {
          switch ( v5 )
          {
            case '#':
            case '$':
            case '%':
            case '&':
            case '\'':
            case '(':
            case ')':
            case '*':
            case '+':
            case ',':
            case '-':
            case '.':
            case '/':
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '8':
            case '<':
            case '=':
            case '>':
            case '?':
            case '@':
            case 'A':
            case 'B':
            case 'C':
            case 'D':
            case 'K':
            case 'L':
              v13 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
              v14 = (_QWORD *)(a1 - v13 * 8);
              if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
                v14 = *(_QWORD **)(a1 - 8);
              v26 = &v14[v13];
              if ( &v14[v13] == v14 )
                return 1;
              v15 = a4 - 1;
              v16 = v14;
              while ( 1 )
              {
                v4 = sub_17ADFE0(*v16, a2, a3, v15);
                if ( !(_BYTE)v4 )
                  break;
                v16 += 3;
                if ( v26 == v16 )
                  return 1;
              }
              return v4;
            case 'T':
              if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
                v17 = *(_QWORD **)(a1 - 8);
              else
                v17 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
              v18 = v17[6];
              if ( *(_BYTE *)(v18 + 16) != 13 )
                return v4;
              v27 = *(_DWORD *)(v18 + 32);
              if ( v27 > 0x40 )
              {
                v24 = v17;
                v25 = v8;
                v23 = v17[6];
                v22 = sub_16A57B0(v18 + 24);
                v8 = v25;
                v17 = v24;
                v19 = -1;
                if ( v27 - v22 <= 0x40 )
                  v19 = **(_DWORD **)(v23 + 24);
              }
              else
              {
                v19 = *(_DWORD *)(v18 + 24);
              }
              if ( !v8 )
                goto LABEL_30;
              v20 = a2;
              v21 = 0;
              break;
            default:
              return v4;
          }
          while ( 1 )
          {
            if ( *v20 == v19 )
            {
              if ( v21 )
                return v4;
              if ( ++v20 == (_DWORD *)v10 )
              {
LABEL_30:
                a1 = *v17;
                --a4;
                v5 = *(unsigned __int8 *)(*v17 + 16LL);
                if ( (unsigned __int8)v5 > 0x10u )
                  goto LABEL_3;
                return 1;
              }
              if ( *v20 == v19 )
                return 0;
              v21 = 1;
            }
            if ( ++v20 == (_DWORD *)v10 )
              goto LABEL_30;
          }
        }
      }
    }
    return 0;
  }
}
