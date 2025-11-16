// Function: sub_A52910
// Address: 0xa52910
//
char __fastcall sub_A52910(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rbx
  int v5; // ecx
  unsigned int v6; // esi
  unsigned __int8 v7; // dl
  int v8; // eax
  __int64 v9; // rdx
  char v10; // dl
  const char *v11; // rsi
  __int64 *v12; // rax
  char v13; // al
  __int64 v14; // r12
  __int64 v15; // r12
  __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-38h]
  char v21; // [rsp+20h] [rbp-30h]

  v3 = *a2;
  if ( (unsigned __int8)v3 > 0x1Cu )
  {
    switch ( (char)v3 )
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
        goto LABEL_6;
      case 'T':
      case 'U':
      case 'V':
        v4 = *((_QWORD *)a2 + 1);
        v5 = *(unsigned __int8 *)(v4 + 8);
        v6 = v5 - 17;
        v7 = *(_BYTE *)(v4 + 8);
        if ( (unsigned int)(v5 - 17) <= 1 )
          v7 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
        if ( v7 <= 3u || v7 == 5 || (v7 & 0xFD) == 4 )
          goto LABEL_6;
        if ( (_BYTE)v5 == 15 )
        {
          if ( (*(_BYTE *)(v4 + 9) & 4) == 0 )
            goto LABEL_10;
          if ( !(unsigned __int8)sub_BCB420(v4) )
            goto LABEL_42;
          v12 = *(__int64 **)(v4 + 16);
          v4 = *v12;
          v5 = *(unsigned __int8 *)(*v12 + 8);
          v6 = v5 - 17;
        }
        else if ( (_BYTE)v5 == 16 )
        {
          do
          {
            v4 = *(_QWORD *)(v4 + 24);
            LOBYTE(v5) = *(_BYTE *)(v4 + 8);
          }
          while ( (_BYTE)v5 == 16 );
          v6 = (unsigned __int8)v5 - 17;
        }
        if ( v6 <= 1 )
          LOBYTE(v5) = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
        if ( (unsigned __int8)v5 > 3u && (_BYTE)v5 != 5 && (v5 & 0xFD) != 4 )
        {
LABEL_42:
          v3 = *a2;
          goto LABEL_9;
        }
LABEL_6:
        v8 = a2[1] >> 1;
        if ( v8 == 127 )
          v8 = -1;
        LODWORD(v17) = v8;
        sub_BB5940(&v17, a1);
        v3 = *a2;
LABEL_9:
        if ( (unsigned __int8)v3 <= 0x1Cu )
          goto LABEL_16;
LABEL_10:
        if ( (unsigned __int8)v3 > 0x36u )
        {
          if ( (unsigned __int8)(v3 - 55) <= 1u )
            goto LABEL_21;
          if ( (_BYTE)v3 == 58 )
          {
            v11 = " disjoint";
            if ( (a2[1] & 2) == 0 )
              return v3;
            goto LABEL_47;
          }
          if ( (_BYTE)v3 == 63 )
            goto LABEL_49;
        }
        else
        {
          v9 = 0x40540000000000LL;
          if ( _bittest64(&v9, v3) )
            goto LABEL_12;
          if ( (unsigned int)(unsigned __int8)v3 - 48 <= 1 )
            goto LABEL_21;
        }
        if ( (((_BYTE)v3 - 68) & 0xFB) == 0 )
        {
          LOBYTE(v3) = sub_B44910(a2);
          v11 = " nneg";
          if ( !(_BYTE)v3 )
            return v3;
          goto LABEL_47;
        }
        if ( (_BYTE)v3 != 67 )
        {
          if ( (_BYTE)v3 == 82 && (a2[1] & 2) != 0 )
          {
            v11 = " samesign";
            goto LABEL_47;
          }
          return v3;
        }
        LOBYTE(v3) = a2[1] >> 1;
        if ( (a2[1] & 2) != 0 )
        {
          sub_904010(a1, " nuw");
          LOBYTE(v3) = a2[1] >> 1;
        }
        if ( (v3 & 2) == 0 )
          return v3;
        break;
      default:
        goto LABEL_9;
    }
    goto LABEL_46;
  }
LABEL_16:
  if ( (_BYTE)v3 != 5 )
    return v3;
  LOWORD(v3) = *((_WORD *)a2 + 1);
  if ( (v3 & 0xFFF7) == 0x11 || (v3 & 0xFFFD) == 0xD )
  {
LABEL_12:
    LOBYTE(v3) = a2[1];
    v10 = (unsigned __int8)v3 >> 1;
    if ( (v3 & 2) != 0 )
    {
      LOBYTE(v3) = sub_904010(a1, " nuw");
      v10 = a2[1] >> 1;
    }
    if ( (v10 & 2) != 0 )
    {
LABEL_46:
      v11 = " nsw";
LABEL_47:
      LOBYTE(v3) = sub_904010(a1, v11);
    }
  }
  else
  {
    if ( (unsigned int)(unsigned __int16)v3 - 19 <= 1 || (unsigned __int16)(v3 - 26) <= 1u )
    {
LABEL_21:
      if ( (a2[1] & 2) == 0 )
        return v3;
      v11 = " exact";
      goto LABEL_47;
    }
    if ( (_WORD)v3 == 34 )
    {
LABEL_49:
      v13 = a2[1] >> 1;
      if ( (a2[1] & 2) != 0 )
      {
        sub_904010(a1, " inbounds");
        v13 = a2[1] >> 1;
      }
      else if ( (v13 & 2) != 0 )
      {
        sub_904010(a1, " nusw");
        v13 = a2[1] >> 1;
      }
      if ( (v13 & 4) != 0 )
        sub_904010(a1, " nuw");
      LOBYTE(v3) = sub_BB52D0(&v17, a2);
      if ( v21 )
      {
        v14 = sub_904010(a1, " inrange(");
        sub_C49420(&v17, v14, 1);
        v15 = sub_904010(v14, ", ");
        sub_C49420(&v19, v15, 1);
        LOBYTE(v3) = sub_904010(v15, ")");
        if ( v21 )
        {
          v21 = 0;
          if ( v20 > 0x40 && v19 )
            LOBYTE(v3) = j_j___libc_free_0_0(v19);
          if ( v18 > 0x40 && v17 )
            LOBYTE(v3) = j_j___libc_free_0_0(v17);
        }
      }
    }
  }
  return v3;
}
