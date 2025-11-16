// Function: sub_10C9850
// Address: 0x10c9850
//
__int64 __fastcall sub_10C9850(__int64 a1, int a2, unsigned __int8 *a3)
{
  char *v5; // r12
  __int64 v6; // rax
  char *v7; // r12
  __int64 v8; // rax
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // rdi
  _BYTE *v16; // rsi
  char v17; // al
  _BYTE *v18; // rsi
  char v19; // al
  _BYTE *v20; // rax
  _BYTE *v21; // rax
  unsigned __int8 *v22; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v23; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v24; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v25; // [rsp-20h] [rbp-20h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = (char *)*((_QWORD *)a3 - 8);
  v6 = *((_QWORD *)v5 + 2);
  if ( !v6 || *(_QWORD *)(v6 + 8) )
    goto LABEL_4;
  v13 = *v5;
  if ( *v5 == 69 )
  {
    v16 = (_BYTE *)*((_QWORD *)v5 - 4);
    if ( *v16 != 56 )
      goto LABEL_4;
    v22 = a3;
    v17 = sub_10B8C30(a1, (__int64)v16);
    a3 = v22;
    if ( v17 )
      goto LABEL_22;
    v13 = *v5;
  }
  if ( v13 != 56 )
    goto LABEL_4;
  v14 = *((_QWORD *)v5 - 8);
  if ( !v14 )
    goto LABEL_4;
  **(_QWORD **)(a1 + 24) = v14;
  v15 = *((_QWORD *)v5 - 4);
  if ( *(_BYTE *)v15 == 17 )
  {
    **(_QWORD **)(a1 + 32) = v15 + 24;
    goto LABEL_22;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v15 + 8) + 8LL) - 17 > 1
    || *(_BYTE *)v15 > 0x15u
    || (v24 = a3, v20 = sub_AD7630(v15, *(unsigned __int8 *)(a1 + 40), (__int64)a3), a3 = v24, !v20)
    || *v20 != 17 )
  {
LABEL_4:
    v7 = (char *)*((_QWORD *)a3 - 4);
    goto LABEL_5;
  }
  **(_QWORD **)(a1 + 32) = v20 + 24;
LABEL_22:
  v7 = (char *)*((_QWORD *)a3 - 4);
  if ( v7 )
  {
    **(_QWORD **)(a1 + 48) = v7;
    return 1;
  }
LABEL_5:
  v8 = *((_QWORD *)v7 + 2);
  if ( !v8 || *(_QWORD *)(v8 + 8) )
    return 0;
  v9 = *v7;
  if ( *v7 == 69 )
  {
    v18 = (_BYTE *)*((_QWORD *)v7 - 4);
    if ( *v18 != 56 )
      return 0;
    v23 = a3;
    v19 = sub_10B8C30(a1, (__int64)v18);
    a3 = v23;
    if ( v19 )
      goto LABEL_14;
    v9 = *v7;
  }
  if ( v9 == 56 )
  {
    v10 = *((_QWORD *)v7 - 8);
    if ( v10 )
    {
      **(_QWORD **)(a1 + 24) = v10;
      v11 = *((_QWORD *)v7 - 4);
      if ( *(_BYTE *)v11 == 17 )
      {
        **(_QWORD **)(a1 + 32) = v11 + 24;
        goto LABEL_14;
      }
      v25 = a3;
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v11 + 8) + 8LL) - 17 <= 1 && *(_BYTE *)v11 <= 0x15u )
      {
        v21 = sub_AD7630(v11, *(unsigned __int8 *)(a1 + 40), (__int64)a3);
        if ( v21 )
        {
          if ( *v21 == 17 )
          {
            a3 = v25;
            **(_QWORD **)(a1 + 32) = v21 + 24;
LABEL_14:
            v12 = *((_QWORD *)a3 - 8);
            if ( v12 )
            {
              **(_QWORD **)(a1 + 48) = v12;
              return 1;
            }
          }
        }
      }
    }
  }
  return 0;
}
