// Function: sub_2ED2890
// Address: 0x2ed2890
//
__int64 __fastcall sub_2ED2890(__int64 a1, int a2, __int64 a3, __int64 a4, _BYTE *a5, _BYTE *a6)
{
  __int64 v6; // r13
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r14
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // edx
  __int64 v19; // rdx

  v6 = (unsigned int)a2;
  v8 = *(_QWORD *)(a1 + 24);
  if ( a2 < 0 )
    v9 = *(_QWORD *)(*(_QWORD *)(v8 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v9 = *(_QWORD *)(*(_QWORD *)(v8 + 304) + 8LL * (unsigned int)a2);
  if ( !v9 )
    return 1;
  if ( (*(_BYTE *)(v9 + 3) & 0x10) != 0 || (*(_BYTE *)(v9 + 4) & 8) != 0 )
  {
    v15 = *(_QWORD *)(v9 + 32);
    v16 = v15;
    if ( v15 )
    {
      while ( (*(_BYTE *)(v16 + 3) & 0x10) != 0 || (*(_BYTE *)(v16 + 4) & 8) != 0 )
      {
        v16 = *(_QWORD *)(v16 + 32);
        if ( !v16 )
          return 1;
      }
      if ( (*(_BYTE *)(v9 + 3) & 0x10) != 0 )
        goto LABEL_30;
      goto LABEL_6;
    }
    return 1;
  }
LABEL_6:
  if ( (*(_BYTE *)(v9 + 4) & 8) == 0 )
  {
LABEL_7:
    v10 = *(_QWORD *)(v9 + 16);
    v11 = sub_2EAB0A0(v9);
    if ( *(_QWORD *)(v10 + 24) == a3
      && (!*(_WORD *)(v10 + 68) || *(_WORD *)(v10 + 68) == 68)
      && *(_QWORD *)(*(_QWORD *)(v10 + 32) + 40LL * (unsigned int)(v11 + 1) + 24) == a4 )
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(v9 + 32);
        if ( !v9 )
          goto LABEL_27;
        while ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 )
        {
          if ( (*(_BYTE *)(v9 + 4) & 8) == 0 )
            goto LABEL_7;
          v9 = *(_QWORD *)(v9 + 32);
          if ( !v9 )
            goto LABEL_27;
        }
      }
    }
    v12 = *(_QWORD *)(a1 + 24);
    if ( (int)v6 < 0 )
      v13 = *(_QWORD *)(*(_QWORD *)(v12 + 56) + 16 * (v6 & 0x7FFFFFFF) + 8);
    else
      v13 = *(_QWORD *)(*(_QWORD *)(v12 + 304) + 8 * v6);
    while ( 1 )
    {
      if ( !v13 )
        return 1;
      if ( (*(_BYTE *)(v13 + 3) & 0x10) == 0 && (*(_BYTE *)(v13 + 4) & 8) == 0 )
        break;
      v13 = *(_QWORD *)(v13 + 32);
    }
    v17 = *(_QWORD *)(v13 + 16);
    v18 = *(unsigned __int16 *)(v17 + 68);
    if ( v18 != 68 )
    {
LABEL_38:
      if ( v18 )
      {
        v19 = *(_QWORD *)(v17 + 24);
        if ( v19 == a4 )
        {
          *a6 = 1;
          return 0;
        }
LABEL_40:
        if ( (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 32), a3, v19) )
        {
          while ( 1 )
          {
            v13 = *(_QWORD *)(v13 + 32);
            if ( !v13 )
              return 1;
            if ( (*(_BYTE *)(v13 + 3) & 0x10) == 0 && (*(_BYTE *)(v13 + 4) & 8) == 0 )
            {
              v17 = *(_QWORD *)(v13 + 16);
              v18 = *(unsigned __int16 *)(v17 + 68);
              if ( v18 != 68 )
                goto LABEL_38;
              goto LABEL_45;
            }
          }
        }
        return 0;
      }
    }
LABEL_45:
    v19 = *(_QWORD *)(*(_QWORD *)(v17 + 32)
                    + 40LL * (-858993459 * (unsigned int)((v13 - *(_QWORD *)(v17 + 32)) >> 3) + 1)
                    + 24);
    goto LABEL_40;
  }
  v15 = *(_QWORD *)(v9 + 32);
LABEL_30:
  v9 = v15;
  if ( !v15 )
  {
LABEL_27:
    *a5 = 1;
    return 1;
  }
  do
  {
    if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 && (*(_BYTE *)(v9 + 4) & 8) == 0 )
      goto LABEL_7;
    v9 = *(_QWORD *)(v9 + 32);
  }
  while ( v9 );
  *a5 = 1;
  return 1;
}
