// Function: sub_B33A00
// Address: 0xb33a00
//
__int64 __fastcall sub_B33A00(__int64 a1, __int64 a2, int a3, int a4, __int64 a5, __int64 a6, __int64 a7, __int64 a8)
{
  unsigned __int64 v9; // rsi
  __int64 v10; // r12
  __int64 v11; // r13
  int v12; // edx
  unsigned int v13; // ecx
  unsigned __int8 v14; // al
  __int64 v15; // rsi
  __int64 *v17; // rax

  v9 = 0;
  if ( a2 )
    v9 = *(_QWORD *)(a2 + 24);
  v10 = sub_B33530((unsigned int **)a1, v9, a2, a3, a4, a5, a7, a8, 0);
  if ( *(_BYTE *)v10 > 0x1Cu )
  {
    switch ( *(_BYTE *)v10 )
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
        goto LABEL_8;
      case 'T':
      case 'U':
      case 'V':
        v11 = *(_QWORD *)(v10 + 8);
        v12 = *(unsigned __int8 *)(v11 + 8);
        v13 = v12 - 17;
        v14 = *(_BYTE *)(v11 + 8);
        if ( (unsigned int)(v12 - 17) <= 1 )
          v14 = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
        if ( v14 <= 3u || v14 == 5 || (v14 & 0xFD) == 4 )
          goto LABEL_8;
        if ( (_BYTE)v12 == 15 )
        {
          if ( (*(_BYTE *)(v11 + 9) & 4) == 0 || !(unsigned __int8)sub_BCB420(*(_QWORD *)(v10 + 8)) )
            return v10;
          v17 = *(__int64 **)(v11 + 16);
          v11 = *v17;
          v12 = *(unsigned __int8 *)(*v17 + 8);
          v13 = v12 - 17;
        }
        else if ( (_BYTE)v12 == 16 )
        {
          do
          {
            v11 = *(_QWORD *)(v11 + 24);
            LOBYTE(v12) = *(_BYTE *)(v11 + 8);
          }
          while ( (_BYTE)v12 == 16 );
          v13 = (unsigned __int8)v12 - 17;
        }
        if ( v13 <= 1 )
          LOBYTE(v12) = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
        if ( (unsigned __int8)v12 <= 3u || (_BYTE)v12 == 5 || (v12 & 0xFD) == 4 )
        {
LABEL_8:
          v15 = *(unsigned int *)(a1 + 104);
          if ( BYTE4(a6) )
            v15 = (unsigned int)a6;
          sub_B45150(v10, v15);
        }
        break;
      default:
        return v10;
    }
  }
  return v10;
}
