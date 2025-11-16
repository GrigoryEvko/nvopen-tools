// Function: sub_F7D9C0
// Address: 0xf7d9c0
//
_BYTE *__fastcall sub_F7D9C0(__int64 a1, unsigned __int8 *a2, __int64 a3, char a4)
{
  int v7; // edx
  _BYTE **v8; // rbx
  _BYTE *result; // rax
  unsigned __int8 *v10; // rax
  _BYTE *v11; // rsi
  __int64 v13; // rsi
  int v14; // r8d
  __int64 *v15; // r14
  __int64 *v16; // rdx
  __int64 *v17; // rcx
  unsigned __int8 v18; // al
  char v19; // al
  __int64 v20; // rax
  unsigned __int8 *v21; // rbx
  __int64 v22; // rdi
  __int64 *v23; // rdx
  __int64 *v24; // [rsp-40h] [rbp-40h]

  if ( a2 == (unsigned __int8 *)a3 )
    return 0;
  v7 = *a2;
  if ( v7 == 63 )
  {
    v13 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
    if ( (a2[7] & 0x40) == 0 )
    {
      v15 = (__int64 *)sub_F79960((__int64)&a2[-v13], (__int64)a2, 1);
      v17 = v16;
      if ( v16 != v15 )
        goto LABEL_19;
      goto LABEL_26;
    }
    v22 = *((_QWORD *)a2 - 1);
    v15 = (__int64 *)sub_F79960(v22, v22 + v13, 1);
    v17 = v23;
    if ( v23 != v15 )
    {
LABEL_19:
      while ( 1 )
      {
        v18 = *(_BYTE *)*v15;
        if ( v18 > 0x15u )
        {
          if ( v18 > 0x1Cu )
          {
            v24 = v17;
            v19 = sub_B19DB0(*(_QWORD *)(*(_QWORD *)a1 + 40LL), *v15, a3);
            v17 = v24;
            if ( !v19 )
              return 0;
          }
          if ( !a4 )
            break;
        }
        v15 += 4;
        if ( v17 == v15 )
          goto LABEL_24;
      }
      v20 = sub_BB5290((__int64)a2);
      if ( !sub_BCAC40(v20, 8) )
        return 0;
LABEL_24:
      if ( (a2[7] & 0x40) == 0 )
      {
        v14 = *((_DWORD *)a2 + 1);
LABEL_26:
        v21 = &a2[-32 * (v14 & 0x7FFFFFF)];
        goto LABEL_27;
      }
      v22 = *((_QWORD *)a2 - 1);
    }
    v21 = (unsigned __int8 *)v22;
    goto LABEL_27;
  }
  if ( (unsigned int)(v7 - 29) <= 0x22 )
  {
    if ( (((_BYTE)v7 - 42) & 0xFD) != 0 )
      return 0;
    if ( (a2[7] & 0x40) != 0 )
    {
      v10 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      v11 = (_BYTE *)*((_QWORD *)v10 + 4);
      if ( *v11 > 0x1Cu )
        goto LABEL_12;
    }
    else
    {
      v10 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v11 = (_BYTE *)*((_QWORD *)v10 + 4);
      if ( *v11 > 0x1Cu )
      {
LABEL_12:
        if ( !(unsigned __int8)sub_B19DB0(*(_QWORD *)(*(_QWORD *)a1 + 40LL), (__int64)v11, a3) )
          return 0;
        if ( (a2[7] & 0x40) != 0 )
          v10 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v10 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      }
    }
    result = *(_BYTE **)v10;
    if ( *result > 0x1Cu )
      return result;
    return 0;
  }
  if ( v7 != 78 )
    return 0;
  if ( (a2[7] & 0x40) != 0 )
  {
    v8 = (_BYTE **)*((_QWORD *)a2 - 1);
    result = *v8;
    if ( **v8 > 0x1Cu )
      return result;
    return 0;
  }
  v21 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
LABEL_27:
  result = *(_BYTE **)v21;
  if ( **(_BYTE **)v21 <= 0x1Cu )
    return 0;
  return result;
}
