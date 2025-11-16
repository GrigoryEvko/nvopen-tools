// Function: sub_5CB890
// Address: 0x5cb890
//
__int64 __fastcall sub_5CB890(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r8
  __int64 v6; // r12
  __int64 v7; // rbx
  char v8; // al
  __int64 v9; // rsi

  if ( *(_BYTE *)(a1 + 8) != *(_BYTE *)(a2 + 8) )
    return 0;
  if ( (_DWORD)a3 || *(_BYTE *)(a1 + 9) == *(_BYTE *)(a2 + 9) )
  {
    v5 = 1;
    v6 = *(_QWORD *)(a2 + 32);
    v7 = *(_QWORD *)(a1 + 32);
    if ( v6 && v7 )
    {
      do
      {
        v8 = *(_BYTE *)(v7 + 10);
        if ( v8 != *(_BYTE *)(v6 + 10) )
          break;
        switch ( v8 )
        {
          case 0:
            break;
          case 1:
          case 2:
            a1 = *(_QWORD *)(v7 + 40);
            v5 = strcmp((const char *)a1, *(const char **)(v6 + 40)) == 0;
            break;
          case 3:
            a1 = *(_QWORD *)(v7 + 40);
            v5 = (unsigned int)sub_73A2C0(a1, *(_QWORD *)(v6 + 40), a3, a4, v5);
            break;
          case 4:
            a1 = *(_QWORD *)(v7 + 40);
            v9 = *(_QWORD *)(v6 + 40);
            v5 = 1;
            if ( a1 != v9 )
              v5 = (unsigned int)sub_8D97D0(a1, v9, 0, a4, 1) != 0;
            break;
          case 5:
            a1 = *(_QWORD *)(v7 + 40);
            v5 = (unsigned int)sub_7386E0(a1, *(_QWORD *)(v6 + 40), 0, a4, v5);
            break;
          default:
            sub_721090(a1);
        }
        v7 = *(_QWORD *)v7;
        v6 = *(_QWORD *)v6;
        if ( !v7 || !v6 )
        {
          if ( !(_DWORD)v5 )
            break;
          goto LABEL_21;
        }
      }
      while ( (_DWORD)v5 );
    }
    else
    {
LABEL_21:
      if ( !(v7 | v6) )
        return (unsigned int)v5;
    }
    LODWORD(v5) = 0;
    return (unsigned int)v5;
  }
  return 0;
}
