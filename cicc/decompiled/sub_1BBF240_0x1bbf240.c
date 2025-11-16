// Function: sub_1BBF240
// Address: 0x1bbf240
//
__int64 __fastcall sub_1BBF240(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, int a6)
{
  unsigned __int8 v8; // al
  __int64 v9; // rax
  _QWORD *v11; // rax
  __int64 v12; // rdi
  _QWORD *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // r15
  _QWORD *v18; // [rsp+8h] [rbp-48h]
  _QWORD v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(_BYTE *)(a1 + 16);
  if ( v8 <= 0x10u )
  {
LABEL_16:
    v15 = *(unsigned int *)(a3 + 8);
    if ( (unsigned int)v15 >= *(_DWORD *)(a3 + 12) )
    {
      sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, (int)a5, a6);
      v15 = *(unsigned int *)(a3 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v15) = a1;
    ++*(_DWORD *)(a3 + 8);
    return 1;
  }
  else
  {
    if ( v8 > 0x17u )
    {
      v9 = *(_QWORD *)(a1 + 8);
      if ( v9 )
      {
        if ( !*(_QWORD *)(v9 + 8) && sub_1A018F0(a2, a1) )
        {
          switch ( *(_BYTE *)(a1 + 16) )
          {
            case '#':
            case '%':
            case '\'':
            case '2':
            case '3':
            case '4':
              if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
                v11 = *(_QWORD **)(a1 - 8);
              else
                v11 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
              if ( !(unsigned __int8)sub_1BBF240(*v11, a2, a3, a4) )
                return 0;
              v12 = *(_QWORD *)(sub_13CF970(a1) + 24);
              goto LABEL_11;
            case '<':
              if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
                v14 = *(_QWORD **)(a1 - 8);
              else
                v14 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
              v19[0] = *v14;
              sub_12A9700(a4, v19);
              goto LABEL_16;
            case '=':
            case '>':
              goto LABEL_16;
            case 'M':
              v16 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
              if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
              {
                a5 = *(_QWORD **)(a1 - 8);
                v18 = &a5[v16];
              }
              else
              {
                v18 = (_QWORD *)a1;
                a5 = (_QWORD *)(a1 - v16 * 8);
              }
              if ( v18 == a5 )
                goto LABEL_16;
              v17 = a5;
              break;
            case 'O':
              if ( !(unsigned __int8)sub_1BBF240(*(_QWORD *)(a1 - 48), a2, a3, a4) )
                return 0;
              v12 = *(_QWORD *)(a1 - 24);
LABEL_11:
              if ( !(unsigned __int8)sub_1BBF240(v12, a2, a3, a4) )
                return 0;
              goto LABEL_16;
            default:
              return 0;
          }
          while ( (unsigned __int8)sub_1BBF240(*v17, a2, a3, a4) )
          {
            v17 += 3;
            if ( v18 == v17 )
              goto LABEL_16;
          }
        }
      }
    }
    return 0;
  }
}
