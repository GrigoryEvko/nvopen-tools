// Function: sub_1653900
// Address: 0x1653900
//
char __fastcall sub_1653900(__int64 a1, __int64 a2)
{
  int v3; // esi
  __int64 v4; // rcx
  __int64 v5; // rcx
  unsigned __int8 *v6; // r14
  unsigned __int8 v7; // dl
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  const char *v10; // rax
  __int64 v11; // r15
  _BYTE *v12; // rax
  char v13; // dl
  unsigned __int8 v14; // dl
  const char *v15; // rax
  _QWORD v17[2]; // [rsp+0h] [rbp-40h] BYREF
  char v18; // [rsp+10h] [rbp-30h]
  char v19; // [rsp+11h] [rbp-2Fh]

  sub_164F0A0(a1, a2);
  v3 = *(unsigned __int16 *)(a2 + 2);
  if ( (_WORD)v3 == 22 )
  {
    v5 = *(unsigned int *)(a2 + 8);
  }
  else
  {
    if ( (unsigned __int16)(v3 - 13) > 0x3Au || (v4 = 0x42005002204800DLL, !_bittest64(&v4, (unsigned int)(v3 - 13))) )
    {
      v19 = 1;
      v15 = "invalid tag";
LABEL_33:
      v17[0] = v15;
      v18 = 3;
      LOBYTE(v8) = sub_16521E0((__int64 *)a1, (__int64)v17);
      if ( *(_QWORD *)a1 )
        LOBYTE(v8) = (unsigned __int8)sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
      return v8;
    }
    v5 = *(unsigned int *)(a2 + 8);
    if ( v3 == 31 )
    {
      v6 = *(unsigned __int8 **)(a2 + 8 * (4 - v5));
      if ( v6 )
      {
        v7 = *v6;
        if ( *v6 > 0xEu )
        {
          if ( (unsigned __int8)(v7 - 32) > 1u )
          {
LABEL_8:
            v19 = 1;
            v17[0] = "invalid pointer to member type";
            v18 = 3;
            LOBYTE(v8) = sub_16521E0((__int64 *)a1, (__int64)v17);
            if ( !*(_QWORD *)a1 )
              return v8;
LABEL_19:
            sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
            LOBYTE(v8) = (unsigned __int8)sub_164ED40((__int64 *)a1, v6);
            return v8;
          }
        }
        else if ( v7 <= 0xAu )
        {
          goto LABEL_8;
        }
      }
    }
  }
  v6 = *(unsigned __int8 **)(a2 + 8 * (1 - v5));
  if ( v6 )
  {
    v9 = *v6;
    if ( *v6 > 0x15u )
    {
      if ( (unsigned __int8)(v9 - 31) > 2u )
        goto LABEL_14;
    }
    else if ( v9 <= 0xAu )
    {
LABEL_14:
      v19 = 1;
      v10 = "invalid scope";
      goto LABEL_15;
    }
  }
  v6 = *(unsigned __int8 **)(a2 + 8 * (3 - v5));
  if ( !v6 )
    goto LABEL_26;
  v14 = *v6;
  if ( *v6 > 0xEu )
  {
    if ( (unsigned __int8)(v14 - 32) > 1u )
      goto LABEL_24;
    goto LABEL_26;
  }
  if ( v14 > 0xAu )
  {
LABEL_26:
    LOBYTE(v8) = v3 != 66;
    if ( v3 == 66 || (unsigned __int16)(*(_WORD *)(a2 + 2) - 15) <= 1u || !*(_BYTE *)(a2 + 56) )
      return v8;
    v19 = 1;
    v15 = "DWARF address space only applies to pointer or reference types";
    goto LABEL_33;
  }
LABEL_24:
  v19 = 1;
  v10 = "invalid base type";
LABEL_15:
  v11 = *(_QWORD *)a1;
  v17[0] = v10;
  v18 = 3;
  if ( !v11 )
  {
    LOBYTE(v8) = *(_BYTE *)(a1 + 74);
    *(_BYTE *)(a1 + 73) = 1;
    *(_BYTE *)(a1 + 72) |= v8;
    return v8;
  }
  sub_16E2CE0(v17, v11);
  v12 = *(_BYTE **)(v11 + 24);
  if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 16) )
  {
    sub_16E7DE0(v11, 10);
  }
  else
  {
    *(_QWORD *)(v11 + 24) = v12 + 1;
    *v12 = 10;
  }
  v8 = *(_QWORD *)a1;
  v13 = *(_BYTE *)(a1 + 74);
  *(_BYTE *)(a1 + 73) = 1;
  *(_BYTE *)(a1 + 72) |= v13;
  if ( v8 )
    goto LABEL_19;
  return v8;
}
