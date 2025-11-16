// Function: sub_B45560
// Address: 0xb45560
//
__int64 __fastcall sub_B45560(unsigned __int8 *a1, unsigned __int64 a2)
{
  unsigned __int8 *v2; // r12
  unsigned __int64 v3; // rax
  __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  char v6; // si
  unsigned __int16 v7; // ax
  int v8; // edx
  __int64 result; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // eax
  unsigned int v14; // r13d
  __int64 v15; // rdx
  int v16; // eax
  int v17; // edx
  _BOOL4 v18; // edx
  int v19; // eax
  _BOOL4 v20; // esi
  bool v21; // dl
  char v22; // si

  v2 = (unsigned __int8 *)a2;
  v3 = *(unsigned __int8 *)a2;
  if ( (unsigned __int8)v3 <= 0x1Cu )
  {
    if ( (_BYTE)v3 != 5 )
      goto LABEL_22;
    v7 = *(_WORD *)(a2 + 2);
    if ( (v7 & 0xFFFD) != 0xD && (v7 & 0xFFF7) != 0x11 )
      goto LABEL_14;
    v11 = *a1;
    if ( (unsigned __int8)v11 > 0x36u )
      goto LABEL_14;
    v12 = 0x40540000000000LL;
    if ( !_bittest64(&v12, v11) )
      goto LABEL_14;
  }
  else
  {
    if ( (unsigned __int8)v3 > 0x36u )
    {
      if ( (_BYTE)v3 != 67 )
        goto LABEL_39;
      goto LABEL_60;
    }
    v4 = 0x40540000000000LL;
    if ( !_bittest64(&v4, v3) )
      goto LABEL_39;
    v5 = *a1;
    if ( (unsigned __int8)v5 > 0x36u || !_bittest64(&v4, v5) )
      goto LABEL_39;
  }
  v6 = 0;
  if ( sub_B44900((__int64)a1) )
    v6 = (v2[1] & 4) != 0;
  sub_B44850(a1, v6);
  a2 = 0;
  if ( sub_B448F0((__int64)a1) )
    a2 = (v2[1] & 2) != 0;
  sub_B447F0(a1, a2);
  LOBYTE(v3) = *v2;
  if ( *v2 == 67 )
  {
LABEL_60:
    LOBYTE(v3) = 67;
    if ( *a1 != 67 )
      goto LABEL_39;
    v22 = 0;
    if ( sub_B44900((__int64)a1) )
      v22 = (v2[1] & 4) != 0;
    sub_B44850(a1, v22);
    a2 = 0;
    if ( sub_B448F0((__int64)a1) )
      a2 = (v2[1] & 2) != 0;
    sub_B447F0(a1, a2);
    LOBYTE(v3) = *v2;
  }
  if ( (unsigned __int8)v3 <= 0x1Cu )
  {
    if ( (_BYTE)v3 != 5 )
      goto LABEL_22;
    v7 = *((_WORD *)v2 + 1);
LABEL_14:
    if ( (unsigned int)v7 - 19 > 1 && (unsigned __int16)(v7 - 26) > 1u )
      goto LABEL_22;
    v8 = *a1;
    if ( (unsigned int)(v8 - 48) > 1 && (unsigned __int8)(v8 - 55) > 1u )
      goto LABEL_22;
    goto LABEL_18;
  }
LABEL_39:
  if ( (unsigned int)(unsigned __int8)v3 - 48 > 1 && (unsigned __int8)(v3 - 55) > 1u )
    goto LABEL_21;
  v17 = *a1;
  if ( (unsigned int)(v17 - 48) <= 1 || (unsigned __int8)(v17 - 55) <= 1u )
  {
LABEL_18:
    a2 = 0;
    if ( sub_B44E60((__int64)a1) )
      a2 = (v2[1] & 2) != 0;
    sub_B448B0((__int64)a1, a2);
LABEL_21:
    if ( *v2 != 58 )
      goto LABEL_22;
    goto LABEL_56;
  }
  if ( *v2 != 58 )
    goto LABEL_22;
LABEL_56:
  if ( *a1 == 58 )
  {
    v21 = 0;
    if ( (a1[1] & 2) != 0 )
      v21 = (v2[1] & 2) != 0;
    a1[1] = a1[1] & 1 | (2 * (v21 | (a1[1] >> 1) & 0xFE));
  }
LABEL_22:
  if ( (unsigned __int8)sub_920620((__int64)v2) && (unsigned __int8)sub_920620((__int64)a1) )
  {
    a2 = (unsigned int)sub_B45210((__int64)a1);
    v13 = v2[1] >> 1;
    if ( v13 != 127 )
      a2 = v13 & (unsigned int)a2;
    sub_B45170((__int64)a1, a2);
    result = *v2;
    v10 = result;
    if ( (_BYTE)result != 63 )
      goto LABEL_25;
  }
  else
  {
    result = *v2;
    v10 = result;
    if ( (_BYTE)result != 63 )
      goto LABEL_25;
  }
  if ( *a1 != 63 )
    goto LABEL_47;
  v14 = sub_B4DE20(a1, a2, v10);
  v16 = sub_B4DE20(v2, a2, v15);
  sub_B4DDE0(a1, v16 & v14);
  result = *v2;
LABEL_25:
  if ( (unsigned __int8)result <= 0x1Cu )
    return result;
  result = (unsigned int)(result - 68);
  if ( (result & 0xFB) == 0 )
  {
    result = (unsigned int)*a1 - 68;
    if ( ((*a1 - 68) & 0xFB) == 0 )
    {
      v20 = 0;
      if ( sub_B44910((__int64)a1) )
        v20 = sub_B44910((__int64)v2);
      result = sub_B448D0((__int64)a1, v20);
    }
  }
  LOBYTE(v10) = *v2;
LABEL_47:
  if ( (_BYTE)v10 == 82 && *a1 == 82 )
  {
    v18 = 0;
    v19 = a1[1];
    LOBYTE(v19) = (unsigned __int8)v19 >> 1;
    if ( (a1[1] & 2) != 0 )
      v18 = (v2[1] & 2) != 0;
    result = a1[1] & 1 | (2 * (v18 | v19 & 0xFFFFFFFE));
    a1[1] = result;
  }
  return result;
}
