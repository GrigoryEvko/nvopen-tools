// Function: sub_100C400
// Address: 0x100c400
//
bool __fastcall sub_100C400(__int64 a1, int a2, unsigned __int8 *a3)
{
  _BYTE *v4; // rax
  unsigned __int8 *v5; // rsi
  int v6; // ecx
  int v7; // ecx
  unsigned __int8 *v8; // rsi
  __int64 v9; // rsi
  unsigned __int8 *v10; // rcx
  int v11; // eax
  int v12; // eax
  unsigned __int8 *v13; // rcx
  __int64 v14; // r12
  unsigned int v15; // r13d
  __int64 v16; // rax
  __int64 v17; // rdx
  _BYTE *v18; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v4 != 44 )
    return 0;
  v5 = (unsigned __int8 *)*((_QWORD *)v4 - 8);
  v6 = *v5;
  if ( (unsigned __int8)v6 > 0x1Cu )
  {
    v7 = v6 - 29;
LABEL_9:
    if ( v7 != 47 )
      return 0;
    v8 = (v5[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v5 - 1) : &v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
    v9 = *(_QWORD *)v8;
    if ( !v9 )
      return 0;
    **(_QWORD **)a1 = v9;
    v10 = (unsigned __int8 *)*((_QWORD *)v4 - 4);
    v11 = *v10;
    if ( (unsigned __int8)v11 > 0x1Cu )
    {
      v12 = v11 - 29;
    }
    else
    {
      if ( (_BYTE)v11 != 5 )
        return 0;
      v12 = *((unsigned __int16 *)v10 + 1);
    }
    if ( v12 != 47 )
      return 0;
    v13 = (v10[7] & 0x40) != 0
        ? (unsigned __int8 *)*((_QWORD *)v10 - 1)
        : &v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
    if ( *(_QWORD *)v13 != *(_QWORD *)(a1 + 8) )
      return 0;
    v14 = *((_QWORD *)a3 - 4);
    if ( !v14 )
      BUG();
    if ( *(_BYTE *)v14 != 17 )
    {
      v17 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v14 + 8) + 8LL) - 17;
      if ( (unsigned int)v17 > 1 )
        return 0;
      if ( *(_BYTE *)v14 > 0x15u )
        return 0;
      v18 = sub_AD7630(v14, 0, v17);
      v14 = (__int64)v18;
      if ( !v18 || *v18 != 17 )
        return 0;
    }
    v15 = *(_DWORD *)(v14 + 32);
    if ( v15 > 0x40 )
    {
      if ( v15 - (unsigned int)sub_C444A0(v14 + 24) > 0x40 )
        return 0;
      v16 = **(_QWORD **)(v14 + 24);
    }
    else
    {
      v16 = *(_QWORD *)(v14 + 24);
    }
    return *(_QWORD *)(a1 + 16) == v16;
  }
  if ( (_BYTE)v6 == 5 )
  {
    v7 = *((unsigned __int16 *)v5 + 1);
    goto LABEL_9;
  }
  return 0;
}
