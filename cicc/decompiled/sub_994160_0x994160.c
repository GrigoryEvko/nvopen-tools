// Function: sub_994160
// Address: 0x994160
//
__int64 __fastcall sub_994160(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 v8; // r14
  __int16 v9; // cx
  __int64 v10; // rbx
  __int64 v11; // r13
  __int16 v12; // ax
  int v13; // eax
  _BYTE *v14; // rax

  if ( *a2 != 86 )
    return 0;
  v3 = *((_QWORD *)a2 - 12);
  if ( *(_BYTE *)v3 != 83 )
    return 0;
  v5 = *((_QWORD *)a2 - 8);
  v6 = *(_QWORD *)(v3 - 64);
  v7 = *((_QWORD *)a2 - 4);
  v8 = *(_QWORD *)(v3 - 32);
  if ( v5 == v6 && v7 == v8 )
  {
    v9 = *(_WORD *)(v3 + 2);
  }
  else
  {
    if ( v5 != v8 || v7 != v6 )
      goto LABEL_21;
    v9 = *(_WORD *)(v3 + 2);
    if ( v5 != v6 )
    {
      if ( (unsigned int)sub_B52870(*(_WORD *)(v3 + 2) & 0x3F) - 4 > 1 || v6 != *(_QWORD *)a1 )
      {
LABEL_17:
        if ( *a2 != 86 )
          return 0;
        v3 = *((_QWORD *)a2 - 12);
        goto LABEL_19;
      }
LABEL_11:
      if ( *(_BYTE *)v8 == 18 )
      {
        **(_QWORD **)(a1 + 8) = v8 + 24;
        return 1;
      }
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17 <= 1 && *(_BYTE *)v8 <= 0x15u )
      {
        v14 = (_BYTE *)sub_AD7630(v8, *(unsigned __int8 *)(a1 + 16));
        if ( v14 )
        {
          if ( *v14 == 18 )
          {
            **(_QWORD **)(a1 + 8) = v14 + 24;
            return 1;
          }
        }
      }
      goto LABEL_17;
    }
  }
  if ( (v9 & 0x3Fu) - 4 <= 1 )
  {
    if ( v6 != *(_QWORD *)a1 )
    {
LABEL_19:
      if ( *(_BYTE *)v3 != 83 )
        return 0;
      v5 = *((_QWORD *)a2 - 8);
      v7 = *((_QWORD *)a2 - 4);
      goto LABEL_21;
    }
    goto LABEL_11;
  }
LABEL_21:
  v10 = *(_QWORD *)(v3 - 64);
  v11 = *(_QWORD *)(v3 - 32);
  if ( v5 == v10 && v7 == v11 )
  {
    v12 = *(_WORD *)(v3 + 2);
    goto LABEL_24;
  }
  if ( v5 == v11 && v7 == v10 )
  {
    v12 = *(_WORD *)(v3 + 2);
    if ( v5 != v10 )
    {
      v13 = sub_B52870(v12 & 0x3F);
      goto LABEL_25;
    }
LABEL_24:
    v13 = v12 & 0x3F;
LABEL_25:
    if ( (unsigned int)(v13 - 12) <= 1 && v10 == *(_QWORD *)(a1 + 24) )
      return sub_9940E0(a1 + 32, v11);
  }
  return 0;
}
