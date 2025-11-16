// Function: sub_13E0700
// Address: 0x13e0700
//
__int64 __fastcall sub_13E0700(__int64 a1, _QWORD *a2, unsigned __int8 *a3, __int64 *a4, int a5)
{
  __int64 v7; // r12
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  void *v12; // rcx
  unsigned __int8 **v13; // rax
  void *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  void *v17; // rcx
  char v18; // [rsp+Bh] [rbp-35h]

  if ( *((_BYTE *)a2 + 16) <= 0x10u && a3[16] <= 0x10u )
  {
    v7 = sub_14D6F90(a1, a2, a3, *a4);
    if ( v7 )
      return v7;
  }
  v7 = sub_13CEFA0(a2, a3, 0);
  if ( v7 )
    return v7;
  v9 = *((unsigned __int8 *)a2 + 16);
  if ( (_DWORD)a1 == 21 )
  {
    if ( (_BYTE)v9 == 45 )
    {
      if ( (unsigned __int8 *)*(a2 - 3) == a3 )
        return (__int64)a2;
    }
    else
    {
      if ( (_BYTE)v9 == 5 )
      {
        v11 = *((unsigned __int16 *)a2 + 9);
        if ( (_WORD)v11 == 21 )
        {
          if ( (unsigned __int8 *)a2[3 * (1LL - (*((_DWORD *)a2 + 5) & 0xFFFFFFF))] == a3 )
            return (__int64)a2;
        }
        else if ( (unsigned __int16)v11 > 0x17u )
        {
          goto LABEL_11;
        }
        v12 = &loc_80A800;
        if ( !_bittest64((const __int64 *)&v12, v11) || (_WORD)v11 != 23 )
          goto LABEL_11;
LABEL_23:
        if ( (*((_BYTE *)a2 + 17) & 4) == 0 )
          goto LABEL_11;
        goto LABEL_24;
      }
      if ( (unsigned __int8)v9 <= 0x17u )
        goto LABEL_11;
      if ( (unsigned __int8)v9 > 0x2Fu )
        goto LABEL_25;
    }
    v15 = 0x80A800000000LL;
    if ( !_bittest64(&v15, v9) || (_BYTE)v9 != 47 )
      goto LABEL_11;
    goto LABEL_23;
  }
  if ( (_BYTE)v9 == 44 )
  {
    if ( (unsigned __int8 *)*(a2 - 3) == a3 )
      return (__int64)a2;
LABEL_36:
    v16 = 0x80A800000000LL;
    if ( !_bittest64(&v16, v9) || (_BYTE)v9 != 47 )
      goto LABEL_11;
    goto LABEL_38;
  }
  if ( (_BYTE)v9 != 5 )
  {
    if ( (unsigned __int8)v9 <= 0x17u )
      goto LABEL_11;
    if ( (unsigned __int8)v9 > 0x2Fu )
      goto LABEL_25;
    goto LABEL_36;
  }
  v10 = *((unsigned __int16 *)a2 + 9);
  if ( (_WORD)v10 == 20 )
  {
    if ( (unsigned __int8 *)a2[3 * (1LL - (*((_DWORD *)a2 + 5) & 0xFFFFFFF))] != a3 )
      goto LABEL_49;
    return (__int64)a2;
  }
  if ( (unsigned __int16)v10 > 0x17u )
    goto LABEL_11;
LABEL_49:
  v17 = &loc_80A800;
  if ( !_bittest64((const __int64 *)&v17, v10) || (_WORD)v10 != 23 )
    goto LABEL_11;
LABEL_38:
  if ( (*((_BYTE *)a2 + 17) & 2) == 0 )
    goto LABEL_11;
LABEL_24:
  v18 = *((_BYTE *)a2 + 16);
  v13 = (unsigned __int8 **)sub_13CF970((__int64)a2);
  LOBYTE(v9) = v18;
  if ( *v13 != a3 )
  {
LABEL_25:
    if ( (_BYTE)v9 == 79 )
    {
LABEL_26:
      v14 = sub_13DF4D0(a1, (unsigned __int8 *)a2, a3, a4, a5);
      if ( v14 )
        return (__int64)v14;
LABEL_12:
      if ( *((_BYTE *)a2 + 16) != 77 && a3[16] != 77 || (v14 = sub_13DF6F0(a1, (unsigned __int8 *)a2, a3, a4, a5)) == 0 )
      {
        if ( (unsigned __int8)sub_13DD0F0((__int64)a2, (__int64)a3, a4, a5, a1 == 21) )
          return (__int64)a2;
        return v7;
      }
      return (__int64)v14;
    }
LABEL_11:
    if ( a3[16] != 79 )
      goto LABEL_12;
    goto LABEL_26;
  }
  return sub_15A06D0(*a2);
}
