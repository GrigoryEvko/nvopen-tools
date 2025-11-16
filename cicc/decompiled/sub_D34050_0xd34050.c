// Function: sub_D34050
// Address: 0xd34050
//
char __fastcall sub_D34050(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 *a4,
        __int64 a5,
        char a6,
        __int64 a7,
        char a8)
{
  __int64 v8; // r15
  char v11; // al
  __int64 v12; // rax
  char result; // al
  __int64 v14; // rax
  char v15; // dl
  bool v16; // al
  unsigned __int8 **v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  bool v20; // al
  __int64 v21; // rax
  __int64 v22; // [rsp-50h] [rbp-50h]
  __int64 v23; // [rsp-50h] [rbp-50h]
  __int64 v24; // [rsp-50h] [rbp-50h]
  __int64 v25; // [rsp-50h] [rbp-50h]

  if ( (*(_BYTE *)(a2 + 28) & 7) != 0 )
    return 1;
  v8 = (__int64)a4;
  if ( !a3 )
    goto LABEL_5;
  v22 = a5;
  v11 = sub_DEF1D0(a1, a3, 1);
  a5 = v22;
  if ( v11 )
    return 1;
  if ( *(_BYTE *)a3 != 63 )
    goto LABEL_5;
  v16 = sub_B4DE40(a3);
  a5 = v22;
  if ( v16 )
  {
    v17 = (unsigned __int8 **)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)));
    if ( (unsigned __int8 **)a3 != v17 )
    {
      a4 = 0;
      do
      {
        if ( **v17 != 17 )
        {
          if ( a4 )
            goto LABEL_31;
          a4 = *v17;
        }
        v17 += 4;
      }
      while ( (unsigned __int8 **)a3 != v17 );
      if ( a4 )
      {
        v18 = *a4;
        if ( (unsigned __int8)v18 <= 0x1Cu )
        {
          if ( (_BYTE)v18 != 5 || (*((_WORD *)a4 + 1) & 0xFFFD) != 0xD && (*((_WORD *)a4 + 1) & 0xFFF7) != 0x11 )
            goto LABEL_31;
        }
        else
        {
          if ( (unsigned __int8)v18 > 0x36u )
            goto LABEL_31;
          v19 = 0x40540000000000LL;
          if ( !_bittest64(&v19, v18) )
            goto LABEL_31;
        }
        if ( (a4[1] & 4) != 0 && **((_BYTE **)a4 - 4) == 17 )
        {
          v21 = sub_DEEF40(a1, *((_QWORD *)a4 - 8));
          a5 = v22;
          if ( *(_WORD *)(v21 + 24) == 8 && v22 == *(_QWORD *)(v21 + 48) && (*(_BYTE *)(v21 + 28) & 4) != 0 )
            return 1;
        }
      }
    }
  }
LABEL_31:
  if ( *(_BYTE *)a3 == 63 )
  {
    v25 = a5;
    v20 = sub_B4DE40(a3);
    a5 = v25;
    if ( v20 )
      return 1;
  }
LABEL_5:
  if ( a8 )
    goto LABEL_6;
  if ( a5 == *(_QWORD *)(a2 + 48) )
  {
    v24 = a5;
    v14 = sub_D33F60((_QWORD *)a2, a5, v8, *(_QWORD *)(a1 + 112), a5);
    a5 = v24;
    if ( v15 )
    {
      a7 = v14;
LABEL_6:
      v23 = a5;
      v12 = sub_D95540(**(_QWORD **)(a2 + 32));
      if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
        v12 = **(_QWORD **)(v12 + 16);
      if ( !sub_B2F070(*(_QWORD *)(**(_QWORD **)(v23 + 32) + 72LL), *(_DWORD *)(v12 + 8) >> 8)
        && ((a7 + 1) & 0xFFFFFFFFFFFFFFFDLL) == 0 )
      {
        return 1;
      }
    }
  }
  result = a6 & (a3 != 0);
  if ( result )
  {
    sub_DF2F80(a1, a3, 1, a4, a5);
    return a6 & (a3 != 0);
  }
  return result;
}
