// Function: sub_643950
// Address: 0x643950
//
__int64 __fastcall sub_643950(__int64 a1, __int64 *a2, __int64 **a3, int a4, __int64 a5)
{
  __int64 i; // r12
  unsigned int v10; // r13d
  __int64 v12; // rax
  __int64 *v13; // rcx
  _QWORD *v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 *j; // rax
  char v21; // dl
  __int64 v22; // rsi

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(i) )
    sub_8AE000(i);
  if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) > 1u )
  {
LABEL_5:
    v10 = 0;
    if ( !a4 )
      sub_685360(2828, a5);
    return v10;
  }
  v12 = *(_QWORD *)(i + 168);
  if ( (*(_BYTE *)(v12 + 109) & 0x20) == 0 )
  {
    v13 = *(__int64 **)(i + 160);
    v14 = *(_QWORD **)v12;
    v15 = v13;
    if ( v13 )
    {
      while ( !*v15 || !v15[1] && (v15[18] & 4) != 0 )
      {
        v15 = (__int64 *)v15[14];
        if ( !v15 )
        {
LABEL_16:
          v13 = 0;
          goto LABEL_17;
        }
      }
      while ( 1 )
      {
        v19 = *v13;
        if ( *v13 )
        {
          if ( v13[1] || (v13[18] & 4) == 0 )
            break;
        }
        v13 = (__int64 *)v13[14];
        if ( !v13 )
          goto LABEL_16;
      }
      if ( !v14 )
        goto LABEL_27;
      do
      {
LABEL_18:
        while ( 1 )
        {
          v16 = v14[5];
          v17 = *(__int64 **)(v16 + 160);
          if ( v17 )
            break;
LABEL_24:
          v14 = (_QWORD *)*v14;
          if ( !v14 )
            goto LABEL_25;
        }
        v18 = *(_QWORD *)(v16 + 160);
        while ( !*(_QWORD *)v18 || !*(_QWORD *)(v18 + 8) && (*(_BYTE *)(v18 + 144) & 4) != 0 )
        {
          v18 = *(_QWORD *)(v18 + 112);
          if ( !v18 )
            goto LABEL_24;
        }
        if ( v13 )
          goto LABEL_5;
        v14 = (_QWORD *)*v14;
        v13 = v17;
      }
      while ( v14 );
LABEL_25:
      if ( v13 )
      {
        v19 = *v13;
LABEL_27:
        for ( j = v13; ; v19 = *j )
        {
          if ( v19 )
          {
            v21 = *((_BYTE *)j + 144);
            if ( j[1] || (v21 & 4) == 0 )
              break;
          }
          j = (__int64 *)j[14];
          if ( !j )
            goto LABEL_33;
        }
        v22 = 1;
LABEL_47:
        if ( (v21 & 0x10) != 0 )
          goto LABEL_5;
        while ( 1 )
        {
          j = (__int64 *)j[14];
          if ( !j )
            break;
          if ( *j )
          {
            v21 = *((_BYTE *)j + 144);
            if ( j[1] || (v21 & 4) == 0 )
            {
              ++v22;
              goto LABEL_47;
            }
          }
        }
LABEL_34:
        *a2 = v22;
        v10 = 1;
        *a3 = v13;
        return v10;
      }
    }
    else
    {
LABEL_17:
      if ( v14 )
        goto LABEL_18;
    }
LABEL_33:
    v22 = 0;
    goto LABEL_34;
  }
  v10 = 0;
  if ( !a4 )
    sub_6851C0(2844, a5);
  return v10;
}
