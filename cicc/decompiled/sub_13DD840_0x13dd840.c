// Function: sub_13DD840
// Address: 0x13dd840
//
__int64 __fastcall sub_13DD840(__int64 a1, _QWORD *a2, _BYTE *a3, __int64 *a4, unsigned int a5)
{
  __int64 v7; // r12
  char v9; // al
  __int16 v10; // dx
  __int16 v11; // cx
  __int64 v12; // rdx
  _BYTE *v13; // rdx
  _BYTE *v14; // rcx
  char v15; // cl
  char v16; // cl
  __int64 v17; // rax
  char v19; // [rsp+1Fh] [rbp-41h] BYREF
  __int64 v20[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( *((_BYTE *)a2 + 16) <= 0x10u && a3[16] <= 0x10u )
  {
    v7 = sub_14D6F90(a1, a2, a3, *a4);
    if ( v7 )
      return v7;
  }
  v7 = sub_13CEFA0(a2, a3, 1);
  if ( v7 )
    return v7;
  v9 = *((_BYTE *)a2 + 16);
  if ( v9 != 39 )
  {
    if ( v9 != 5 )
      goto LABEL_24;
    v10 = *((_WORD *)a2 + 9);
    v11 = v10;
    if ( v10 != 15 )
    {
      if ( (_DWORD)a1 != 18 )
        goto LABEL_10;
LABEL_47:
      if ( v11 != 21 )
        goto LABEL_18;
      goto LABEL_48;
    }
    v13 = (_BYTE *)a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
    v14 = (_BYTE *)a2[3 * (1LL - (*((_DWORD *)a2 + 5) & 0xFFFFFFF))];
    if ( !v13 )
      goto LABEL_30;
LABEL_29:
    if ( v14 == a3 )
      goto LABEL_33;
    goto LABEL_30;
  }
  v13 = (_BYTE *)*(a2 - 6);
  v14 = (_BYTE *)*(a2 - 3);
  if ( v13 )
    goto LABEL_29;
LABEL_30:
  if ( !v14 || v13 != a3 )
    goto LABEL_18;
  v13 = v14;
LABEL_33:
  if ( (_DWORD)a1 == 18 )
  {
    if ( ((*((_BYTE *)a2 + 17) >> 1) & 2) == 0 )
    {
      v16 = v13[16];
      if ( v16 != 42 )
      {
        if ( v16 == 5
          && *((_WORD *)v13 + 9) == 18
          && *(_BYTE **)&v13[24 * (1LL - (*((_DWORD *)v13 + 5) & 0xFFFFFFF))] == a3 )
        {
          return (__int64)v13;
        }
        goto LABEL_44;
      }
      if ( *((_BYTE **)v13 - 3) != a3 )
      {
LABEL_44:
        if ( v9 != 45 )
        {
          if ( v9 != 5 )
            goto LABEL_17;
          v11 = *((_WORD *)a2 + 9);
          goto LABEL_47;
        }
LABEL_26:
        if ( (_BYTE *)*(a2 - 3) == a3 )
          return sub_15A06D0(*a2);
        goto LABEL_18;
      }
    }
    return (__int64)v13;
  }
  if ( (*((_BYTE *)a2 + 17) & 2) != 0 )
    return (__int64)v13;
  v15 = v13[16];
  if ( v15 == 41 )
  {
    if ( *((_BYTE **)v13 - 3) != a3 )
      goto LABEL_24;
    return (__int64)v13;
  }
  if ( v15 == 5 && *((_WORD *)v13 + 9) == 17 && *(_BYTE **)&v13[24 * (1LL - (*((_DWORD *)v13 + 5) & 0xFFFFFFF))] == a3 )
    return (__int64)v13;
LABEL_24:
  if ( (_DWORD)a1 == 18 )
    goto LABEL_44;
  if ( v9 == 44 )
    goto LABEL_26;
  if ( v9 == 5 )
  {
    v10 = *((_WORD *)a2 + 9);
LABEL_10:
    if ( v10 != 20 )
    {
      if ( v10 == 17 )
      {
        if ( a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)] )
        {
          v12 = a2[3 * (1LL - (*((_DWORD *)a2 + 5) & 0xFFFFFFF))];
          if ( *(_BYTE *)(v12 + 16) == 13 )
            goto LABEL_14;
        }
      }
LABEL_18:
      if ( a3[16] != 79 )
        goto LABEL_19;
      goto LABEL_52;
    }
LABEL_48:
    if ( (_BYTE *)a2[3 * (1LL - (*((_DWORD *)a2 + 5) & 0xFFFFFFF))] == a3 )
      return sub_15A06D0(*a2);
    goto LABEL_18;
  }
  if ( v9 != 41 )
    goto LABEL_17;
  if ( !*(a2 - 6) )
    goto LABEL_18;
  v12 = *(a2 - 3);
  if ( *(_BYTE *)(v12 + 16) != 13 )
    goto LABEL_18;
LABEL_14:
  if ( a3[16] == 13 )
  {
    sub_16AA580(v20, v12 + 24, a3 + 24, &v19);
    sub_135E100(v20);
    if ( v19 )
      return sub_15A06D0(*a2);
    v9 = *((_BYTE *)a2 + 16);
  }
LABEL_17:
  if ( v9 != 79 )
    goto LABEL_18;
LABEL_52:
  v17 = sub_13DF4D0((unsigned int)a1, a2, a3, a4, a5);
  if ( v17 )
    return v17;
  v9 = *((_BYTE *)a2 + 16);
LABEL_19:
  if ( v9 == 77 || a3[16] == 77 )
  {
    v17 = sub_13DF6F0((unsigned int)a1, a2, a3, a4, a5);
    if ( v17 )
      return v17;
  }
  if ( !(unsigned __int8)sub_13DD0F0((__int64)a2, (__int64)a3, a4, a5, a1 == 18) )
    return v7;
  return sub_15A06D0(*a2);
}
