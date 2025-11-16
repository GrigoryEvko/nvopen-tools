// Function: sub_B45260
// Address: 0xb45260
//
__int64 __fastcall sub_B45260(unsigned __int8 *a1, __int64 a2, char a3)
{
  unsigned __int8 *v3; // rbx
  unsigned __int64 v4; // rax
  __int64 v5; // rcx
  unsigned __int64 v6; // rax
  int v7; // edx
  __int64 result; // rax
  __int64 v9; // rdx
  int v10; // eax
  int v11; // eax
  unsigned int v12; // r13d
  __int64 v13; // rdx
  int v14; // eax
  int v15; // edx
  _BOOL4 v16; // ecx
  bool v17; // al

  v3 = (unsigned __int8 *)a2;
  if ( a3 )
  {
    v4 = *a1;
    if ( (unsigned __int8)v4 <= 0x36u )
    {
      v5 = 0x40540000000000LL;
      if ( _bittest64(&v5, v4) )
      {
        v6 = *(unsigned __int8 *)a2;
        if ( (unsigned __int8)v6 <= 0x1Cu )
        {
          if ( (_BYTE)v6 != 5 )
            goto LABEL_13;
          v10 = *(unsigned __int16 *)(a2 + 2);
          if ( (*(_WORD *)(a2 + 2) & 0xFFFD) != 0xD && (v10 & 0xFFF7) != 0x11 )
            goto LABEL_24;
        }
        else
        {
          if ( (unsigned __int8)v6 > 0x36u )
          {
            if ( (_BYTE)v6 != 67 )
              goto LABEL_7;
LABEL_48:
            LOBYTE(v6) = 67;
LABEL_8:
            if ( (unsigned __int8)(v6 - 55) > 1u )
              goto LABEL_12;
            goto LABEL_9;
          }
          if ( !_bittest64(&v5, v6) )
            goto LABEL_7;
        }
        sub_B44850(a1, (*(_BYTE *)(a2 + 1) & 4) != 0);
        a2 = (*(_BYTE *)(a2 + 1) & 2) != 0;
        sub_B447F0(a1, a2);
      }
    }
  }
  LOBYTE(v6) = *v3;
  if ( *v3 == 67 )
  {
    if ( *a1 != 67 )
      goto LABEL_48;
    sub_B44850(a1, (v3[1] & 4) != 0);
    a2 = (v3[1] & 2) != 0;
    sub_B447F0(a1, a2);
    LOBYTE(v6) = *v3;
  }
  if ( (unsigned __int8)v6 <= 0x1Cu )
  {
    if ( (_BYTE)v6 != 5 )
      goto LABEL_13;
    v10 = *((unsigned __int16 *)v3 + 1);
LABEL_24:
    if ( (unsigned __int16)(v10 - 26) > 1u && (unsigned int)(v10 - 19) > 1 )
      goto LABEL_13;
    v11 = *a1;
    if ( (unsigned __int8)(v11 - 55) > 1u && (unsigned int)(v11 - 48) > 1 )
      goto LABEL_13;
    goto LABEL_11;
  }
LABEL_7:
  if ( (unsigned int)(unsigned __int8)v6 - 48 > 1 )
    goto LABEL_8;
LABEL_9:
  v7 = *a1;
  if ( (unsigned int)(v7 - 48) <= 1 || (unsigned __int8)(v7 - 55) <= 1u )
  {
LABEL_11:
    a2 = (v3[1] & 2) != 0;
    sub_B448B0((__int64)a1, a2);
  }
LABEL_12:
  if ( *v3 == 58 && *a1 == 58 )
    a1[1] = a1[1] & 1 | (2 * (((v3[1] & 2) != 0) | (a1[1] >> 1) & 0xFE));
LABEL_13:
  if ( (unsigned __int8)sub_920620((__int64)v3) && (unsigned __int8)sub_920620((__int64)a1) )
  {
    a2 = v3[1] >> 1;
    if ( (_DWORD)a2 == 127 )
      a2 = 0xFFFFFFFFLL;
    sub_B45170((__int64)a1, a2);
    result = *v3;
    v9 = result;
    if ( (_BYTE)result != 63 )
      goto LABEL_16;
  }
  else
  {
    result = *v3;
    v9 = result;
    if ( (_BYTE)result != 63 )
      goto LABEL_16;
  }
  if ( *a1 != 63 )
    goto LABEL_36;
  v12 = sub_B4DE20(a1, a2, v9);
  v14 = sub_B4DE20(v3, a2, v13);
  sub_B4DDE0(a1, v14 | v12);
  result = *v3;
LABEL_16:
  if ( (unsigned __int8)result <= 0x1Cu )
    return result;
  result = (unsigned int)(result - 68);
  if ( (result & 0xFB) == 0 )
  {
    result = (unsigned int)*a1 - 68;
    if ( ((*a1 - 68) & 0xFB) == 0 )
    {
      v17 = sub_B44910((__int64)v3);
      result = sub_B448D0((__int64)a1, v17);
    }
  }
  LOBYTE(v9) = *v3;
LABEL_36:
  if ( (_BYTE)v9 == 82 && *a1 == 82 )
  {
    v15 = a1[1] & 1;
    v16 = (v3[1] & 2) != 0;
    result = v15 | (2 * (v16 | (a1[1] >> 1) & 0xFEu));
    a1[1] = v15 | (2 * (v16 | (a1[1] >> 1) & 0xFE));
  }
  return result;
}
