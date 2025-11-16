// Function: sub_2CBFC80
// Address: 0x2cbfc80
//
__int64 __fastcall sub_2CBFC80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 *a10,
        _QWORD *a11)
{
  _QWORD *v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v14; // rsi
  __int16 v15; // cx
  __int64 v16; // rax
  unsigned int v17; // r12d
  __int16 v19; // cx
  unsigned __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rcx

  v11 = (_QWORD *)(a2 + 48);
  v12 = *v11 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v12 == v11 )
    goto LABEL_38;
  if ( !v12 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v12 - 24) - 30 > 0xA )
LABEL_38:
    BUG();
  v14 = *(_QWORD *)(v12 - 120);
  v15 = *(_WORD *)(v14 + 2);
  if ( (((v15 & 0x3F) - 36) & 0xFFFB) != 0 )
  {
    v19 = v15 & 0x3B;
    if ( v19 == 34 )
    {
      if ( *(_QWORD *)(v12 - 56) != a3 )
        goto LABEL_6;
    }
    else
    {
      if ( v19 == 35 )
      {
        if ( a7 != *(_QWORD *)(v12 - 56) )
          goto LABEL_6;
LABEL_17:
        *a10 = *(_QWORD *)(v14 - 64);
        *a11 = *(_QWORD *)(v14 - 32);
        goto LABEL_6;
      }
      if ( (((*(_WORD *)(v14 + 2) & 0x3F) - 37) & 0xFFFB) != 0 )
        return 0;
      if ( a7 != *(_QWORD *)(v12 - 56) )
        goto LABEL_6;
    }
    *a11 = *(_QWORD *)(v14 - 64);
    v16 = *(_QWORD *)(v14 - 32);
    *a10 = v16;
    goto LABEL_7;
  }
  if ( a3 == *(_QWORD *)(v12 - 56) )
    goto LABEL_17;
LABEL_6:
  v16 = *a10;
LABEL_7:
  if ( !v16 || *a11 == 0 || *a11 != a9 || a8 != v16 )
    return 0;
  v20 = *(_QWORD *)(a5 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v20 == a5 + 48 )
    goto LABEL_40;
  if ( !v20 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v20 - 24) - 30 > 0xA )
LABEL_40:
    BUG();
  v21 = *(_QWORD *)(v20 - 120);
  v22 = *(unsigned __int16 *)(v21 + 2);
  v23 = *(_WORD *)(v21 + 2) & 0x3F;
  if ( (((*(_WORD *)(v21 + 2) & 0x3F) - 36) & 0xFFFB) == 0 )
  {
    if ( a4 != *(_QWORD *)(v20 - 56) )
      return 1;
LABEL_33:
    if ( !(unsigned __int8)sub_D48480(a1, *(_QWORD *)(v21 - 64), v22, v23) )
      sub_D48480(a1, *(_QWORD *)(v21 - 32), v26, v27);
    return 1;
  }
  v22 &= 0x3Bu;
  if ( (_WORD)v22 == 34 )
  {
    if ( a4 != *(_QWORD *)(v20 - 56) )
      return 1;
    goto LABEL_29;
  }
  if ( (_WORD)v22 == 35 )
  {
    if ( a6 != *(_QWORD *)(v20 - 56) )
      return 1;
    goto LABEL_33;
  }
  v23 = (unsigned int)(v23 - 37);
  if ( (v23 & 0xFFFB) != 0 )
    return 0;
  if ( a6 != *(_QWORD *)(v20 - 56) )
    return 1;
LABEL_29:
  v17 = sub_D48480(a1, *(_QWORD *)(v21 - 64), v22, v23);
  if ( (_BYTE)v17 )
  {
    sub_D48480(a1, *(_QWORD *)(v21 - 32), v24, v25);
    return v17;
  }
  return 1;
}
