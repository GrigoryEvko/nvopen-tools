// Function: sub_11DAC90
// Address: 0x11dac90
//
__int64 __fastcall sub_11DAC90(char *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rcx
  char *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  char *v9; // rcx
  unsigned __int8 v10; // dl
  char *v12; // rsi
  unsigned __int8 v13; // dl
  unsigned __int8 v14; // dl
  unsigned __int8 v15; // dl
  unsigned __int8 v16; // dl
  unsigned __int8 v17; // dl
  unsigned __int8 v18; // dl

  v5 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
  if ( (a1[7] & 0x40) != 0 )
  {
    v6 = (char *)*((_QWORD *)a1 - 1);
    a1 = &v6[v5];
  }
  else
  {
    v6 = &a1[-v5];
  }
  v7 = v5 >> 5;
  v8 = v5 >> 7;
  if ( v8 )
  {
    v9 = &v6[128 * v8];
    while ( 1 )
    {
      v10 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v6 + 8LL) + 8LL);
      if ( v10 <= 3u || v10 == 5 || (v10 & 0xFD) == 4 )
        goto LABEL_6;
      v12 = v6 + 32;
      v13 = *(_BYTE *)(*(_QWORD *)(*((_QWORD *)v6 + 4) + 8LL) + 8LL);
      if ( v13 <= 3u || v13 == 5 || (v13 & 0xFD) == 4 )
      {
LABEL_13:
        LOBYTE(a5) = a1 != v12;
        return a5;
      }
      v14 = *(_BYTE *)(*(_QWORD *)(*((_QWORD *)v6 + 8) + 8LL) + 8LL);
      if ( v14 <= 3u || v14 == 5 || (v14 & 0xFD) == 4 )
      {
        LOBYTE(a5) = v6 + 64 != a1;
        return a5;
      }
      v12 = v6 + 96;
      v15 = *(_BYTE *)(*(_QWORD *)(*((_QWORD *)v6 + 12) + 8LL) + 8LL);
      if ( v15 <= 3u || v15 == 5 || (v15 & 0xFD) == 4 )
        goto LABEL_13;
      v6 += 128;
      if ( v9 == v6 )
      {
        v7 = (a1 - v6) >> 5;
        break;
      }
    }
  }
  if ( v7 == 2 )
  {
LABEL_35:
    v18 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v6 + 8LL) + 8LL);
    if ( v18 > 3u && v18 != 5 && (v18 & 0xFD) != 4 )
    {
      v6 += 32;
LABEL_26:
      v16 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v6 + 8LL) + 8LL);
      if ( v16 > 3u && v16 != 5 )
      {
        a5 = 0;
        if ( (v16 & 0xFD) != 4 )
          return a5;
      }
      goto LABEL_6;
    }
    goto LABEL_6;
  }
  if ( v7 != 3 )
  {
    a5 = 0;
    if ( v7 != 1 )
      return a5;
    goto LABEL_26;
  }
  v17 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v6 + 8LL) + 8LL);
  if ( v17 > 3u && v17 != 5 && (v17 & 0xFD) != 4 )
  {
    v6 += 32;
    goto LABEL_35;
  }
LABEL_6:
  LOBYTE(a5) = a1 != v6;
  return a5;
}
