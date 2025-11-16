// Function: sub_89AB40
// Address: 0x89ab40
//
_BOOL8 __fastcall sub_89AB40(__int64 a1, __int64 a2, __int16 a3, __int64 a4, _UNKNOWN *__ptr32 *a5)
{
  int v5; // r15d
  int v9; // r13d
  char v10; // dl
  __int128 v11; // rdi
  char v12; // r9
  char v13; // r10
  __int128 v15; // rdi
  __m128i *v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rax
  char v20; // dl
  __int64 v21; // rax
  char v22; // dl
  int v23; // [rsp+8h] [rbp-58h]
  int v24; // [rsp+Ch] [rbp-54h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  __m128i *v26; // [rsp+10h] [rbp-50h]
  int v27; // [rsp+20h] [rbp-40h]
  unsigned int v28; // [rsp+24h] [rbp-3Ch]
  unsigned int v29; // [rsp+28h] [rbp-38h]
  unsigned int v30; // [rsp+2Ch] [rbp-34h]

  v5 = a3 & 0x20;
  v28 = a3 & 1;
  v23 = a3 & 4;
  v24 = a3 & 8;
  v30 = a3 & 0x80;
  v27 = a3 & 0x10;
  if ( (a3 & 0x10) != 0 )
  {
    if ( (a3 & 0x80) != 0 )
      v29 = 324;
    else
      v29 = 16724;
    v30 = 6;
  }
  else if ( (a3 & 0x80) != 0 )
  {
    v30 = 0;
    v29 = 4;
  }
  else
  {
    v29 = 16404;
  }
  if ( (a3 & 0x40) != 0 )
  {
    v29 |= 0x100u;
    v30 |= 4u;
  }
  v9 = 1;
  while ( 1 )
  {
    if ( !v5 )
    {
      while ( 1 )
      {
        if ( !a1 )
        {
          if ( a2 )
            goto LABEL_37;
          goto LABEL_40;
        }
        if ( *(_BYTE *)(a1 + 8) != 3 )
          break;
        a1 = *(_QWORD *)a1;
      }
      while ( a2 )
      {
LABEL_37:
        if ( *(_BYTE *)(a2 + 8) != 3 )
          goto LABEL_12;
        a2 = *(_QWORD *)a2;
      }
LABEL_40:
      if ( v9 )
        return (a1 | a2) == 0;
      return 0;
    }
LABEL_12:
    if ( !a1 || !a2 )
      goto LABEL_40;
    v10 = *(_BYTE *)(a1 + 8);
    if ( v10 != *(_BYTE *)(a2 + 8) )
      return 0;
    if ( ((*(_BYTE *)(a2 + 24) ^ *(_BYTE *)(a1 + 24)) & 0x10) != 0 && (a3 & 0x400) == 0 )
    {
      v9 = 0;
      goto LABEL_10;
    }
    if ( v10 != 1 )
      break;
    *(_QWORD *)&v15 = *(_QWORD *)(a1 + 32);
    *((_QWORD *)&v15 + 1) = *(_QWORD *)(a2 + 32);
    if ( v15 == 0 )
      goto LABEL_31;
    if ( !(_QWORD)v15 || !*((_QWORD *)&v15 + 1) )
      return 0;
    if ( (_QWORD)v15 == *((_QWORD *)&v15 + 1) )
      goto LABEL_31;
    v26 = *(__m128i **)(a2 + 32);
    if ( (unsigned int)sub_739430(v15, *((__int64 *)&v15 + 1), v30, a4, a5) )
      goto LABEL_31;
    v16 = v26;
    if ( !v24 || *(_BYTE *)(v15 + 173) != 12 )
      goto LABEL_49;
    if ( *(_BYTE *)(v15 + 176) )
    {
      a4 = v28;
      if ( !v28 )
        return 0;
LABEL_51:
      if ( v16[10].m128i_i8[13] )
        return 0;
      goto LABEL_31;
    }
    v16 = v26;
    if ( !(unsigned int)sub_88F430(v15, v26) )
    {
LABEL_49:
      if ( !v28 )
        return 0;
      if ( *(_BYTE *)(v15 + 173) )
        goto LABEL_51;
    }
LABEL_31:
    if ( !v9 )
      return 0;
    v9 = 1;
LABEL_10:
    a1 = *(_QWORD *)a1;
    a2 = *(_QWORD *)a2;
  }
  if ( !v10 )
  {
    *(_QWORD *)&v11 = *(_QWORD *)(a1 + 32);
    *((_QWORD *)&v11 + 1) = *(_QWORD *)(a2 + 32);
    if ( v11 != 0 )
    {
      if ( !(_QWORD)v11 || !*((_QWORD *)&v11 + 1) )
        return 0;
      if ( (_QWORD)v11 != *((_QWORD *)&v11 + 1) )
      {
        v25 = *(_QWORD *)(a2 + 32);
        if ( !(unsigned int)sub_8D97D0(v11, *((_QWORD *)&v11 + 1), v29, a4, a5) )
        {
          *((_QWORD *)&v11 + 1) = v25;
          v12 = *(_BYTE *)(v11 + 140);
          if ( !v28 )
          {
            v13 = *(_BYTE *)(v25 + 140);
LABEL_25:
            if ( v12 == 12 )
            {
              do
                *(_QWORD *)&v11 = *(_QWORD *)(v11 + 160);
              while ( *(_BYTE *)(v11 + 140) == 12 );
            }
            if ( v13 == 12 )
            {
              do
                *((_QWORD *)&v11 + 1) = *(_QWORD *)(*((_QWORD *)&v11 + 1) + 160LL);
              while ( *(_BYTE *)(*((_QWORD *)&v11 + 1) + 140LL) == 12 );
            }
            if ( !v23 || !(unsigned int)sub_8D97D0(v11, *((_QWORD *)&v11 + 1), v29, a4, a5) )
              return 0;
            goto LABEL_31;
          }
          if ( v12 == 12 )
          {
            v19 = v11;
            do
            {
              v19 = *(_QWORD *)(v19 + 160);
              v20 = *(_BYTE *)(v19 + 140);
            }
            while ( v20 == 12 );
          }
          else
          {
            v20 = *(_BYTE *)(v11 + 140);
          }
          if ( v20 )
          {
            v13 = *(_BYTE *)(v25 + 140);
            if ( v13 == 12 )
            {
              v21 = v25;
              do
              {
                v21 = *(_QWORD *)(v21 + 160);
                v22 = *(_BYTE *)(v21 + 140);
              }
              while ( v22 == 12 );
            }
            else
            {
              v22 = *(_BYTE *)(v25 + 140);
            }
            if ( v22 )
              goto LABEL_25;
          }
        }
      }
    }
    goto LABEL_31;
  }
  if ( v10 != 2 )
    goto LABEL_10;
  v17 = *(_QWORD *)(a1 + 32);
  v18 = *(_QWORD *)(a2 + 32);
  if ( v17 == v18 || (unsigned int)sub_89AAB0(v17, v18, v27 != 0) )
    goto LABEL_31;
  return 0;
}
