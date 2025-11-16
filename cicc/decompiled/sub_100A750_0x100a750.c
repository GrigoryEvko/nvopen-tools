// Function: sub_100A750
// Address: 0x100a750
//
_BOOL8 __fastcall sub_100A750(__int64 a1, int a2, unsigned __int8 *a3)
{
  _BOOL4 v3; // r12d
  unsigned __int8 *v5; // r13
  unsigned __int64 v7; // rax
  __int64 v8; // rsi
  int v9; // eax
  _BOOL4 v10; // r12d
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned __int8 *v13; // r13
  unsigned __int64 v14; // rax
  __int64 v15; // rsi
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  _BYTE *v20; // rax
  _BYTE *v21; // rax
  unsigned __int8 *v22; // [rsp+8h] [rbp-28h]
  unsigned __int8 *v23; // [rsp+8h] [rbp-28h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
  v7 = *v5;
  if ( (unsigned __int8)v7 <= 0x1Cu )
  {
    if ( (_BYTE)v7 != 5 )
      goto LABEL_16;
    v9 = *((unsigned __int16 *)v5 + 1);
    if ( (*((_WORD *)v5 + 1) & 0xFFF7) != 0x11 && (v9 & 0xFFFD) != 0xD )
      goto LABEL_16;
  }
  else
  {
    if ( (unsigned __int8)v7 > 0x36u )
      goto LABEL_16;
    v8 = 0x40540000000000LL;
    if ( !_bittest64(&v8, v7) )
      goto LABEL_16;
    v9 = (unsigned __int8)v7 - 29;
  }
  if ( v9 != 25 )
    goto LABEL_16;
  v10 = (v5[1] & 2) != 0;
  if ( (v5[1] & 2) == 0 )
    goto LABEL_16;
  v11 = *((_QWORD *)v5 - 8);
  if ( !v11 )
    goto LABEL_16;
  **(_QWORD **)a1 = v11;
  v12 = *((_QWORD *)v5 - 4);
  if ( *(_BYTE *)v12 == 17 )
  {
    **(_QWORD **)(a1 + 8) = v12 + 24;
    goto LABEL_13;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v12 + 8) + 8LL) - 17 > 1
    || *(_BYTE *)v12 > 0x15u
    || (v22 = a3, v20 = sub_AD7630(v12, *(unsigned __int8 *)(a1 + 16), (__int64)a3), a3 = v22, !v20)
    || *v20 != 17 )
  {
LABEL_16:
    v13 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
LABEL_17:
    v14 = *v13;
    if ( (unsigned __int8)v14 <= 0x1Cu )
    {
      if ( (_BYTE)v14 != 5 )
        return 0;
      v16 = *((unsigned __int16 *)v13 + 1);
      if ( (*((_WORD *)v13 + 1) & 0xFFF7) != 0x11 && (v16 & 0xFFFD) != 0xD )
        return 0;
    }
    else
    {
      if ( (unsigned __int8)v14 > 0x36u )
        return 0;
      v15 = 0x40540000000000LL;
      if ( !_bittest64(&v15, v14) )
        return 0;
      v16 = (unsigned __int8)v14 - 29;
    }
    if ( v16 == 25 )
    {
      v3 = (v13[1] & 2) != 0;
      if ( (v13[1] & 2) != 0 )
      {
        v17 = *((_QWORD *)v13 - 8);
        if ( v17 )
        {
          **(_QWORD **)a1 = v17;
          v18 = *((_QWORD *)v13 - 4);
          if ( *(_BYTE *)v18 == 17 )
          {
            **(_QWORD **)(a1 + 8) = v18 + 24;
          }
          else
          {
            v23 = a3;
            if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v18 + 8) + 8LL) - 17 > 1 )
              return 0;
            if ( *(_BYTE *)v18 > 0x15u )
              return 0;
            v21 = sub_AD7630(v18, *(unsigned __int8 *)(a1 + 16), (__int64)a3);
            if ( !v21 || *v21 != 17 )
              return 0;
            a3 = v23;
            **(_QWORD **)(a1 + 8) = v21 + 24;
          }
          **(_QWORD **)(a1 + 24) = v13;
          v19 = *((_QWORD *)a3 - 8);
          if ( v19 )
          {
            **(_QWORD **)(a1 + 32) = v19;
            return v3;
          }
        }
      }
    }
    return 0;
  }
  **(_QWORD **)(a1 + 8) = v20 + 24;
LABEL_13:
  **(_QWORD **)(a1 + 24) = v5;
  v13 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
  if ( !v13 )
    goto LABEL_17;
  **(_QWORD **)(a1 + 32) = v13;
  return v10;
}
