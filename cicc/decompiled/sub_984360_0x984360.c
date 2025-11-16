// Function: sub_984360
// Address: 0x984360
//
__int64 __fastcall sub_984360(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3, unsigned int *a4)
{
  int v4; // eax
  __int64 v6; // r12
  __int64 v9; // r14
  int v10; // edx
  __int64 v11; // rax
  unsigned int v12; // r8d
  __int64 v13; // r15
  char v14; // al
  __int64 v15; // rax
  char v16; // al
  unsigned __int8 *v17; // rax

  v4 = *a2;
  if ( (unsigned __int8)v4 <= 0x1Cu || (unsigned int)(v4 - 67) > 0xC )
    return 0;
  *a4 = v4 - 29;
  v9 = *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL);
  v10 = *a3;
  if ( (unsigned __int8)v10 > 0x1Cu )
  {
    if ( (unsigned int)(v10 - 67) <= 0xC )
    {
      if ( v4 == v10 )
      {
        v6 = *((_QWORD *)a3 - 4);
        if ( v9 == *(_QWORD *)(v6 + 8) )
          return v6;
      }
      return 0;
    }
  }
  else if ( (unsigned __int8)v10 <= 0x15u )
  {
    v11 = sub_B43CC0();
    v12 = *a4;
    v13 = v11;
    switch ( *a4 )
    {
      case '&':
        v6 = *(_QWORD *)(a1 - 32);
        if ( !v6 )
          BUG();
        if ( *(_BYTE *)v6 <= 0x15u && v9 == *(_QWORD *)(v6 + 8) )
          goto LABEL_18;
        v16 = sub_B532B0(*(_WORD *)(a1 + 2) & 0x3F);
        v6 = sub_96F480(39 - ((unsigned int)(v16 == 0) - 1), (__int64)a3, v9, v13);
        goto LABEL_16;
      case '\'':
        if ( !(unsigned __int8)sub_B532A0(*(_WORD *)(a1 + 2) & 0x3F) )
          return 0;
        v6 = sub_AD4C30(a3, v9, 0);
        goto LABEL_16;
      case '(':
        if ( !(unsigned __int8)sub_B532B0(*(_WORD *)(a1 + 2) & 0x3F) )
          return 0;
        v6 = sub_AD4C30(a3, v9, 1);
        goto LABEL_16;
      case ')':
        v6 = sub_96F480(0x2Bu, (__int64)a3, v9, v11);
        goto LABEL_16;
      case '*':
        v6 = sub_96F480(0x2Cu, (__int64)a3, v9, v11);
        goto LABEL_16;
      case '+':
        v6 = sub_96F480(0x29u, (__int64)a3, v9, v11);
        goto LABEL_16;
      case ',':
        v6 = sub_96F480(0x2Au, (__int64)a3, v9, v11);
        goto LABEL_16;
      case '-':
        v6 = sub_96F480(0x2Eu, (__int64)a3, v9, v11);
        goto LABEL_16;
      case '.':
        v6 = sub_96F480(0x2Du, (__int64)a3, v9, v11);
LABEL_16:
        if ( !v6 )
          return 0;
        v12 = *a4;
LABEL_18:
        v15 = sub_96F480(v12, v6, *((_QWORD *)a3 + 1), v13);
        if ( v15 )
        {
          if ( a3 != (unsigned __int8 *)v15 )
            return 0;
        }
        return v6;
      default:
        return 0;
    }
  }
  if ( v4 != 67 )
    return 0;
  v6 = *(_QWORD *)(a1 - 32);
  v14 = *(_BYTE *)v6;
  if ( *(_BYTE *)v6 <= 0x1Cu || v14 != 68 && v14 != 69 )
    return 0;
  v17 = *(unsigned __int8 **)(v6 - 32);
  if ( !v17 || a3 != v17 )
    return 0;
  return v6;
}
