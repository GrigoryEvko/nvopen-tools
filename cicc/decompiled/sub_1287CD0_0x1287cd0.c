// Function: sub_1287CD0
// Address: 0x1287cd0
//
__int64 __fastcall sub_1287CD0(
        __int64 a1,
        __int64 *a2,
        _DWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        _BYTE *a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v15; // rax
  _QWORD *v16; // r8
  _QWORD *v17; // rax
  _QWORD *v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rdx
  unsigned __int64 v22; // rax
  __int64 v24; // rax
  __int64 v25; // rcx
  _QWORD *v26; // rax

  if ( (_DWORD)a7 )
  {
    if ( (_DWORD)a7 != 1 )
    {
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = 0;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    }
    sub_1284570(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
    return a1;
  }
  else
  {
    v15 = a2[4];
    v16 = (_QWORD *)(v15 + 544);
    v17 = *(_QWORD **)(v15 + 552);
    if ( !v17 )
      goto LABEL_9;
    v18 = v16;
    do
    {
      while ( 1 )
      {
        v19 = v17[2];
        v20 = v17[3];
        if ( v17[4] >= (unsigned __int64)a8 )
          break;
        v17 = (_QWORD *)v17[3];
        if ( !v20 )
          goto LABEL_7;
      }
      v18 = v17;
      v17 = (_QWORD *)v17[2];
    }
    while ( v19 );
LABEL_7:
    if ( v16 != v18 && v18[4] <= (unsigned __int64)a8 )
    {
      v24 = sub_1287B50((__int64)a2, a8, (__int64)a3, v19);
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_QWORD *)a1 = v24;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    }
    else
    {
LABEL_9:
      v21 = *(_QWORD *)a8;
      v22 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)a8 + 24LL) + 8LL);
      if ( (unsigned __int8)v22 > 0x10u )
      {
        if ( *(_BYTE *)(v21 + 8) != 15 )
          goto LABEL_11;
        goto LABEL_21;
      }
      v25 = 100990;
      if ( _bittest64(&v25, v22) )
      {
        v26 = sub_12810A0(a2, (unsigned __int64)a8, a9, a12 & 1);
        *(_BYTE *)(a1 + 12) &= ~1u;
        *(_QWORD *)a1 = v26;
        *(_DWORD *)(a1 + 8) = 0;
        *(_DWORD *)(a1 + 16) = 0;
        return a1;
      }
      if ( *(_BYTE *)(v21 + 8) != 15 )
LABEL_11:
        sub_127B550("unexpected error generating l-value!", a3, 1);
      if ( (_BYTE)v22 != 12 )
LABEL_21:
        sub_127B550("unexpected error generating l-value!", a3, 1);
      *(_QWORD *)a1 = a8;
      *(_BYTE *)(a1 + 12) &= ~1u;
      *(_DWORD *)(a1 + 8) = 0;
      *(_DWORD *)(a1 + 16) = 0;
      return a1;
    }
  }
}
