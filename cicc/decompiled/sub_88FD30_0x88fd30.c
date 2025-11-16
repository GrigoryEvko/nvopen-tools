// Function: sub_88FD30
// Address: 0x88fd30
//
__int64 __fastcall sub_88FD30(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r14
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v8; // rdi
  char v9; // dl
  __int16 v10; // si
  __int64 v11; // rsi
  __int64 v12; // r11
  __int64 v13; // rdx
  __int64 *v14; // rsi
  __int64 *v15; // rax
  __int64 *v16; // r8
  unsigned int v17; // edi
  __int64 v18; // r12
  __int64 *v19; // r9
  __int64 *i; // rsi
  __int64 *v21; // rdi
  _QWORD *v22; // [rsp+0h] [rbp-40h]
  _QWORD *v23; // [rsp+0h] [rbp-40h]

  if ( !a2 )
    return 0;
  v3 = a2;
  if ( *(_QWORD *)(a2 + 32) )
    goto LABEL_3;
  v15 = *(__int64 **)(a1 + 8);
  if ( !v15 )
    return 0;
  v12 = 0;
  v16 = 0;
  v13 = 0;
  while ( 1 )
  {
    if ( v3 && *((_DWORD *)v15 + 7) == *(_DWORD *)(v3 + 24) )
    {
      v18 = *(_QWORD *)v3;
      *(_QWORD *)v3 = 0;
      if ( v13 )
      {
        v19 = 0;
        for ( i = (__int64 *)v13; *((_DWORD *)i + 7) < *(_DWORD *)(v3 + 28); i = (__int64 *)*i )
        {
          v21 = (__int64 *)*i;
          v19 = i;
          if ( !*i )
            goto LABEL_47;
        }
        if ( !v19 )
          goto LABEL_51;
        v21 = (__int64 *)*v19;
        i = v19;
LABEL_47:
        *(_QWORD *)v3 = v21;
        *i = v3;
      }
      else
      {
LABEL_51:
        *(_QWORD *)v3 = v13;
        v13 = v3;
      }
      *(_QWORD *)(v3 + 32) = v16;
      v3 = v18;
      if ( *((_WORD *)v15 + 12) == 9 )
        goto LABEL_38;
    }
    else
    {
      if ( !v13 )
        goto LABEL_23;
      if ( *((_WORD *)v15 + 12) == 9 )
      {
LABEL_38:
        v14 = (__int64 *)*v15;
        if ( !*v15 )
          goto LABEL_25;
        goto LABEL_33;
      }
    }
    v17 = *(_DWORD *)(v13 + 28);
    if ( *((_DWORD *)v15 + 7) == v17 )
      goto LABEL_22;
    v14 = (__int64 *)*v15;
    if ( !*v15 )
      break;
    if ( v17 < *((_DWORD *)v14 + 7) || !v17 )
      goto LABEL_22;
LABEL_33:
    v16 = v15;
    v15 = v14;
  }
  if ( v17 )
    goto LABEL_25;
LABEL_22:
  v11 = *(_QWORD *)v13;
  *(_QWORD *)(v13 + 40) = v15;
  *(_QWORD *)v13 = v12;
  v12 = v13;
  v13 = v11;
LABEL_23:
  v14 = (__int64 *)*v15;
  if ( *v15 && v3 | v13 )
    goto LABEL_33;
LABEL_25:
  v3 = v12;
  if ( v12 )
  {
LABEL_3:
    v5 = 0;
    while ( 1 )
    {
      v6 = v3;
      v3 = *(_QWORD *)v3;
      if ( !*(_DWORD *)(v6 + 28) )
        goto LABEL_6;
      if ( *(_BYTE *)(v6 + 48) )
      {
LABEL_12:
        sub_878A90(v6);
        if ( !v3 )
          return v5;
      }
      else
      {
        if ( !*(_BYTE *)(v6 + 49) )
        {
          if ( !*(_BYTE *)(v6 + 51) )
          {
            v8 = *(_QWORD *)(v6 + 8);
            v9 = *(_BYTE *)(v8 + 80);
            switch ( v9 )
            {
              case 4:
              case 5:
              case 6:
                if ( !sub_879510((_QWORD *)v8) )
                {
                  v23 = **(_QWORD ***)(v6 + 32);
                  sub_88F240(v6, 17);
                  sub_7AEA30(v23, a1, *(_QWORD *)(v6 + 16));
                }
                goto LABEL_12;
              case 8:
              case 9:
              case 21:
                goto LABEL_14;
              case 10:
              case 19:
              case 20:
                v10 = 75;
                if ( v9 == 19 )
                  v10 = 17;
                v22 = **(_QWORD ***)(v6 + 32);
                sub_88F240(v6, v10);
                sub_7AE1D0(v22);
                goto LABEL_12;
              default:
                sub_721090();
            }
          }
          if ( *(_BYTE *)(v6 + 52) )
            goto LABEL_12;
LABEL_14:
          sub_88F310(v6);
          goto LABEL_12;
        }
        if ( !a3 )
          goto LABEL_14;
        *(_QWORD *)v6 = v5;
        v5 = v6;
LABEL_6:
        if ( !v3 )
          return v5;
      }
    }
  }
  return 0;
}
