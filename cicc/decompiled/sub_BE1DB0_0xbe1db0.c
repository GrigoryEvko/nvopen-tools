// Function: sub_BE1DB0
// Address: 0xbe1db0
//
unsigned __int64 __fastcall sub_BE1DB0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int8 v3; // al
  __int64 v4; // rdx
  unsigned __int8 *v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  const char *v8; // r14
  const char *v9; // rax
  bool v10; // zf
  unsigned __int8 v11; // al
  bool v12; // dl
  __int64 v13; // rcx
  unsigned __int8 *v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 result; // rax
  __int64 v18; // rdx
  unsigned __int8 *v19; // rdx
  unsigned __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r15
  __int64 v23; // r14
  int v24; // ecx
  const char *v25; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v26[4]; // [rsp+10h] [rbp-50h] BYREF
  char v27; // [rsp+30h] [rbp-30h]
  char v28; // [rsp+31h] [rbp-2Fh]

  sub_BDAAE0((__int64)a1, (const char *)a2);
  if ( (unsigned __int16)sub_AF18C0(a2) != 22
    && (unsigned __int16)sub_AF18C0(a2) != 15
    && (unsigned __int16)sub_AF18C0(a2) != 31
    && (unsigned __int16)sub_AF18C0(a2) != 16
    && (unsigned __int16)sub_AF18C0(a2) != 66
    && (unsigned __int16)sub_AF18C0(a2) != 38
    && (unsigned __int16)sub_AF18C0(a2) != 75
    && (unsigned __int16)sub_AF18C0(a2) != 53
    && (unsigned __int16)sub_AF18C0(a2) != 55
    && (unsigned __int16)sub_AF18C0(a2) != 71
    && (unsigned __int16)sub_AF18C0(a2) != 17152
    && (unsigned __int16)sub_AF18C0(a2) != 13
    && ((unsigned __int16)sub_AF18C0(a2) != 52 || (*(_BYTE *)(a2 + 21) & 0x10) == 0)
    && (unsigned __int16)sub_AF18C0(a2) != 28
    && (unsigned __int16)sub_AF18C0(a2) != 42
    && (unsigned __int16)sub_AF18C0(a2) != 32
    && (unsigned __int16)sub_AF18C0(a2) != 67 )
  {
    v25 = (const char *)a2;
    v28 = 1;
    v26[0] = "invalid tag";
    v27 = 3;
    return sub_BE1CC0(a1, (__int64)v26, &v25);
  }
  v2 = a2 - 16;
  if ( (unsigned __int16)sub_AF18C0(a2) == 31 )
  {
    v3 = *(_BYTE *)(a2 - 16);
    v4 = (v3 & 2) != 0 ? *(_QWORD *)(a2 - 32) : v2 - 8LL * ((v3 >> 2) & 0xF);
    v5 = *(unsigned __int8 **)(v4 + 32);
    if ( v5 )
    {
      v6 = *v5;
      if ( (unsigned __int8)v6 > 0x24u || (v7 = 0x140000F000LL, !_bittest64(&v7, v6)) )
      {
        v8 = (const char *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 4);
        v28 = 1;
        v9 = "invalid pointer to member type";
        goto LABEL_22;
      }
    }
  }
  v10 = (unsigned __int16)sub_AF18C0(a2) == 32;
  v11 = *(_BYTE *)(a2 - 16);
  if ( !v10 )
  {
    v12 = (v11 & 2) != 0;
LABEL_12:
    if ( v12 )
    {
      v13 = *(_QWORD *)(a2 - 32);
LABEL_14:
      v14 = *(unsigned __int8 **)(v13 + 8);
      if ( !v14 )
        goto LABEL_18;
      v15 = *v14;
      if ( (unsigned __int8)v15 <= 0x24u )
      {
        v16 = 0x16007FF000LL;
        if ( _bittest64(&v16, v15) )
        {
          v13 = *(_QWORD *)(a2 - 32);
          goto LABEL_18;
        }
      }
      v23 = *(_QWORD *)(a2 - 32);
      goto LABEL_64;
    }
    v13 = v2 - 8LL * ((v11 >> 2) & 0xF);
LABEL_27:
    v19 = *(unsigned __int8 **)(v13 + 8);
    if ( !v19 )
    {
LABEL_18:
      result = *(_QWORD *)(v13 + 24);
      if ( result )
      {
LABEL_19:
        result = *(unsigned __int8 *)result;
        if ( (unsigned __int8)result > 0x24u || (v18 = 0x140000F000LL, !_bittest64(&v18, result)) )
        {
          v8 = (const char *)*((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 3);
          v28 = 1;
          v9 = "invalid base type";
          goto LABEL_22;
        }
      }
LABEL_31:
      if ( *(_BYTE *)(a2 + 48) )
      {
        result = sub_AF18C0(a2);
        if ( (_WORD)result != 15 )
        {
          result = sub_AF18C0(a2);
          if ( (_WORD)result != 16 )
          {
            result = sub_AF18C0(a2);
            if ( (_WORD)result != 66 )
            {
              v28 = 1;
              v26[0] = "DWARF address space only applies to pointer or reference types";
              v27 = 3;
              result = sub_BDD6D0(a1, (__int64)v26);
              if ( *a1 )
                return (unsigned __int64)sub_BD9900(a1, (const char *)a2);
            }
          }
        }
      }
      return result;
    }
    v20 = *v19;
    if ( (unsigned __int8)v20 <= 0x24u )
    {
      v21 = 0x16007FF000LL;
      if ( _bittest64(&v21, v20) )
      {
        result = *(_QWORD *)(v2 - 8LL * ((v11 >> 2) & 0xF) + 24);
        if ( result )
          goto LABEL_19;
        goto LABEL_31;
      }
    }
    v23 = v2 - 8LL * ((v11 >> 2) & 0xF);
LABEL_64:
    v8 = *(const char **)(v23 + 8);
    v9 = "invalid scope";
    v28 = 1;
LABEL_22:
    v26[0] = v9;
    v27 = 3;
    result = sub_BDD6D0(a1, (__int64)v26);
    if ( *a1 )
    {
      result = (unsigned __int64)sub_BD9900(a1, (const char *)a2);
      if ( v8 )
        return (unsigned __int64)sub_BD9900(a1, v8);
    }
    return result;
  }
  v12 = (v11 & 2) != 0;
  if ( (v11 & 2) != 0 )
  {
    v13 = *(_QWORD *)(a2 - 32);
    v22 = *(_QWORD *)(v13 + 24);
    if ( !v22 )
      goto LABEL_14;
  }
  else
  {
    v13 = v2 - 8LL * ((v11 >> 2) & 0xF);
    v22 = *(_QWORD *)(v13 + 24);
    if ( !v22 )
      goto LABEL_27;
  }
  if ( *(_BYTE *)v22 == 14 )
  {
    if ( (unsigned __int16)sub_AF18C0(v22) == 4 )
    {
      v11 = *(_BYTE *)(a2 - 16);
      v12 = (v11 & 2) != 0;
      goto LABEL_12;
    }
  }
  else if ( *(_BYTE *)v22 == 12 )
  {
    v24 = *(_DWORD *)(v22 + 44);
    if ( (unsigned int)(v24 - 5) <= 3 || v24 == 2 )
      goto LABEL_12;
  }
  v28 = 1;
  v26[0] = "invalid set base type";
  v27 = 3;
  result = sub_BDD6D0(a1, (__int64)v26);
  if ( *a1 )
  {
    sub_BD9900(a1, (const char *)a2);
    return (unsigned __int64)sub_BD9900(a1, (const char *)v22);
  }
  return result;
}
