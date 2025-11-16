// Function: sub_99AC20
// Address: 0x99ac20
//
unsigned __int64 __fastcall sub_99AC20(
        __int64 a1,
        __int64 a2,
        unsigned __int8 *a3,
        __int64 *a4,
        __int64 *a5,
        unsigned int *a6)
{
  __int64 v7; // r15
  unsigned int v8; // r12d
  unsigned int v9; // r8d
  unsigned int v10; // r14d
  __int64 v12; // r9
  __int64 v14; // rsi
  int v15; // edx
  unsigned int v16; // ecx
  unsigned __int8 v17; // al
  unsigned int v18; // eax
  __int64 *v19; // rax
  __int64 v20; // r8
  unsigned int v21; // [rsp+0h] [rbp-80h]
  __int64 v24; // [rsp+18h] [rbp-68h]

  v7 = *(_QWORD *)(a1 - 64);
  v24 = *(_QWORD *)(a1 - 32);
  v8 = *(_WORD *)(a1 + 2) & 0x3F;
  if ( *(_BYTE *)a1 <= 0x1Cu )
  {
LABEL_3:
    v9 = *(_WORD *)(a1 + 2) & 0x3F;
    v10 = 0;
  }
  else
  {
    switch ( *(_BYTE *)a1 )
    {
      case ')':
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'J':
      case 'K':
      case 'S':
        goto LABEL_17;
      case 'T':
      case 'U':
      case 'V':
        v14 = *(_QWORD *)(a1 + 8);
        v15 = *(unsigned __int8 *)(v14 + 8);
        v16 = v15 - 17;
        v17 = *(_BYTE *)(v14 + 8);
        if ( (unsigned int)(v15 - 17) <= 1 )
          v17 = *(_BYTE *)(**(_QWORD **)(v14 + 16) + 8LL);
        if ( v17 <= 3u || v17 == 5 )
        {
LABEL_17:
          v9 = *(_WORD *)(a1 + 2) & 0x3F;
LABEL_18:
          v21 = v9;
          v18 = sub_B45210(a1);
          v9 = v21;
          v10 = v18;
          break;
        }
        v9 = *(_WORD *)(a1 + 2) & 0x3F;
        if ( (v17 & 0xFD) == 4 )
          goto LABEL_18;
        if ( (_BYTE)v15 == 15 )
        {
          v10 = 0;
          if ( (*(_BYTE *)(v14 + 9) & 4) == 0 )
            break;
          if ( !(unsigned __int8)sub_BCB420(*(_QWORD *)(a1 + 8)) )
          {
            v9 = *(_WORD *)(a1 + 2) & 0x3F;
            break;
          }
          v19 = *(__int64 **)(v14 + 16);
          v14 = *v19;
          v9 = *(_WORD *)(a1 + 2) & 0x3F;
          v15 = *(unsigned __int8 *)(*v19 + 8);
          v16 = v15 - 17;
        }
        else if ( (_BYTE)v15 == 16 )
        {
          do
          {
            v14 = *(_QWORD *)(v14 + 24);
            LOBYTE(v15) = *(_BYTE *)(v14 + 8);
          }
          while ( (_BYTE)v15 == 16 );
          v9 = *(_WORD *)(a1 + 2) & 0x3F;
          v16 = (unsigned __int8)v15 - 17;
        }
        if ( v16 <= 1 )
          LOBYTE(v15) = *(_BYTE *)(**(_QWORD **)(v14 + 16) + 8LL);
        if ( (unsigned __int8)v15 <= 3u )
          goto LABEL_18;
        if ( (_BYTE)v15 == 5 )
          goto LABEL_18;
        v10 = 0;
        if ( (v15 & 0xFD) == 4 )
          goto LABEL_18;
        break;
      default:
        goto LABEL_3;
    }
  }
  if ( !(unsigned __int8)sub_B52830(v9) )
  {
    if ( a6 && *(_QWORD *)(a2 + 8) != *(_QWORD *)(v7 + 8) )
    {
      v12 = sub_984360(a1, (unsigned __int8 *)a2, a3, a6);
      if ( v12 )
      {
        if ( *a6 - 41 < 2 )
          v10 |= 8u;
        return sub_9989E0(v8, v10, v7, v24, *(_QWORD *)(a2 - 32), v12, a4, a5);
      }
      v20 = sub_984360(a1, a3, (unsigned __int8 *)a2, a6);
      if ( v20 )
      {
        if ( *a6 - 41 < 2 )
          v10 |= 8u;
        return sub_9989E0(v8, v10, v7, v24, v20, *((_QWORD *)a3 - 4), a4, a5);
      }
    }
    return sub_9989E0(v8, v10, v7, v24, a2, (__int64)a3, a4, a5);
  }
  return 0;
}
