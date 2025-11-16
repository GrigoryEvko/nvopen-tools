// Function: sub_6E2250
// Address: 0x6e2250
//
__int64 __fastcall sub_6E2250(__int64 a1, __int64 *a2, unsigned __int8 a3, int a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  __int64 v8; // r14
  __int64 v11; // rbx
  unsigned __int8 v12; // di
  int v13; // edx
  __int64 result; // rax
  __int64 v15; // rax
  char v16; // cl
  __int64 v17; // rax
  _QWORD **v18; // rdx
  _QWORD *i; // rax
  __int64 *v20; // [rsp+8h] [rbp-38h]

  v6 = a2;
  v8 = a5;
  v11 = a6;
  if ( a6 )
  {
    if ( a5 )
      goto LABEL_34;
  }
  else if ( a5 )
  {
    v11 = a5 + 136;
    if ( !a4 )
    {
LABEL_4:
      *a2 = 0;
      goto LABEL_5;
    }
LABEL_21:
    sub_6E1DD0(a2);
    v6 = a2;
    if ( (*(_BYTE *)(v11 + 40) & 4) == 0 )
    {
LABEL_6:
      v12 = a3;
      if ( (*(_BYTE *)(v11 + 42) & 0x10) != 0 )
        v12 = 5;
LABEL_8:
      v20 = v6;
      sub_6E1E00(v12, a1, 0, 0);
      if ( a4 )
      {
        sub_6E2170(*v20);
        *(_BYTE *)(v11 + 42) &= ~2u;
      }
      if ( (*(_BYTE *)(v11 + 40) & 2) != 0 )
      {
        *(_WORD *)(a1 + 18) |= 0x408u;
      }
      else if ( dword_4A52070[0] )
      {
        *(_BYTE *)(a1 + 18) |= 8u;
      }
      v13 = *(_BYTE *)(v11 + 42) & 0x40;
      result = v13 | *(_BYTE *)(a1 + 19) & 0xBFu;
      *(_BYTE *)(a1 + 19) = v13 | *(_BYTE *)(a1 + 19) & 0xBF;
      if ( (*(_BYTE *)(v11 + 42) & 0x20) == 0 )
        *(_BYTE *)(a1 + 17) |= 2u;
      goto LABEL_14;
    }
LABEL_22:
    v12 = 3;
    if ( a3 <= 3u )
      v12 = a3;
    goto LABEL_8;
  }
  if ( a6 )
  {
    v8 = *(_QWORD *)(a6 + 16);
    if ( !a4 )
      goto LABEL_4;
    goto LABEL_21;
  }
LABEL_34:
  if ( !a4 )
  {
    *a2 = 0;
    if ( !a6 )
    {
      sub_6E1E00(a3, a1, 0, 0);
      goto LABEL_37;
    }
LABEL_5:
    if ( (*(_BYTE *)(v11 + 40) & 4) == 0 )
      goto LABEL_6;
    goto LABEL_22;
  }
  sub_6E1DD0(a2);
  v6 = a2;
  if ( v11 )
    goto LABEL_5;
  sub_6E1E00(a3, a1, 0, 0);
  sub_6E2170(*a2);
LABEL_37:
  result = (__int64)dword_4A52070;
  if ( dword_4A52070[0] )
    *(_BYTE *)(a1 + 18) |= 8u;
LABEL_14:
  if ( v8 )
  {
    v15 = *(_QWORD *)v8;
    if ( *(_QWORD *)v8 )
    {
      v16 = *(_BYTE *)(v15 + 80);
      if ( (unsigned __int8)(v16 - 8) <= 1u )
      {
        v17 = *(_QWORD *)(v15 + 88);
        v18 = (_QWORD **)(v17 + 160);
        if ( v16 == 9 )
          v18 = (_QWORD **)(v17 + 208);
        for ( i = *v18; i; i = (_QWORD *)*i )
          v18 = (_QWORD **)i;
        *(_QWORD *)(a1 + 120) = v18;
        if ( *(_BYTE *)(*(_QWORD *)v8 + 80LL) == 8 )
          *(_BYTE *)(qword_4D03C50 + 18LL) |= 8u;
      }
    }
    return sub_6E2220(v8);
  }
  return result;
}
