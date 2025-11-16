// Function: sub_AA8C90
// Address: 0xaa8c90
//
__int64 __fastcall sub_AA8C90(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int8 v3; // al
  unsigned __int8 v4; // bl
  __int64 v6; // rcx
  int v7; // esi
  unsigned int v8; // eax
  __int64 result; // rax
  char *v10; // r15
  char v11; // dl
  _QWORD *i; // r14
  _BYTE *v13; // rax
  int v14; // eax
  __int64 v15; // rsi
  char v16; // bl
  char *v17; // r14
  char v18; // al
  int v19; // [rsp-40h] [rbp-40h]
  char v20; // [rsp-39h] [rbp-39h]
  char v21; // [rsp-39h] [rbp-39h]

  if ( a1 == a2 )
    return 32;
  v2 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v2 + 8) != 14 )
    return 42;
  v3 = *(_BYTE *)a1;
  v4 = *(_BYTE *)a2;
  if ( *(_BYTE *)a1 == 5 )
  {
    if ( v4 == 5 || v4 <= 3u )
    {
LABEL_23:
      if ( *(_WORD *)(a1 + 2) != 34 )
        return 42;
      v10 = *(char **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      if ( v4 != 20 )
      {
        if ( v4 > 3u )
        {
          if ( v4 != 63 && (v4 != 5 || *(_WORD *)(a2 + 2) != 34) )
            return 42;
          v16 = *v10;
          if ( (unsigned __int8)*v10 > 3u )
            return 42;
          v17 = *(char **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
          v21 = *v17;
          if ( (unsigned __int8)*v17 > 3u )
            return 42;
          if ( v10 == v17 )
            return 42;
          if ( !(unsigned __int8)sub_AA8C10(a1) )
            return 42;
          v18 = sub_AA8C10(a2);
          if ( v21 == 1 || v16 == 1 || !v18 )
            return 42;
          v15 = (__int64)v17;
          goto LABEL_60;
        }
        v11 = *v10;
        if ( (unsigned __int8)*v10 <= 3u && v10 != (char *)a2 )
        {
          for ( i = (_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))); (_QWORD *)a1 != i; i += 4 )
          {
            v13 = (_BYTE *)*i;
            if ( *(_BYTE *)*i != 17 )
              return 42;
            if ( *((_DWORD *)v13 + 8) <= 0x40u )
            {
              if ( *((_QWORD *)v13 + 3) )
                return 42;
            }
            else
            {
              v19 = *((_DWORD *)v13 + 8);
              v20 = v11;
              v14 = sub_C444A0(v13 + 24);
              v11 = v20;
              if ( v19 != v14 )
                return 42;
            }
          }
          if ( v11 != 1 )
          {
            v15 = a2;
            if ( v4 != 1 )
            {
LABEL_60:
              a1 = (__int64)v10;
              return sub_AA8900(a1, v15);
            }
          }
        }
        return 42;
      }
      if ( (unsigned __int8)*v10 > 3u || (v10[32] & 0xF) == 9 || (*(_BYTE *)(a1 + 1) & 2) == 0 )
        return 42;
      return 34;
    }
    v6 = 3;
  }
  else
  {
    v6 = 2;
    if ( v3 > 3u )
    {
      v6 = v3 == 4;
      if ( v4 == 5 )
      {
LABEL_9:
        v8 = sub_AA8C90(a2, a1, v2, v6);
        if ( v8 != 42 )
          return sub_B52F50(v8);
        return 42;
      }
    }
    else if ( v4 == 5 )
    {
      goto LABEL_9;
    }
    if ( v4 <= 3u )
    {
      v7 = 2;
      goto LABEL_8;
    }
  }
  v7 = 1;
  if ( v4 != 4 )
  {
    if ( v3 == 4 )
      goto LABEL_11;
    if ( v3 <= 3u )
    {
      if ( v4 != 20
        || (*(_BYTE *)(a1 + 32) & 0xF) == 9
        || v3 == 1
        || (unsigned __int8)sub_B2F070(0, *(_DWORD *)(v2 + 8) >> 8) )
      {
        return 42;
      }
      return 34;
    }
    goto LABEL_22;
  }
LABEL_8:
  if ( v7 > (int)v6 )
    goto LABEL_9;
  if ( v3 != 4 )
  {
    if ( v3 <= 3u )
    {
      if ( v4 > 3u )
        return 33;
      if ( v3 == 1 || v4 == 1 )
        return 42;
      v15 = a2;
      return sub_AA8900(a1, v15);
    }
LABEL_22:
    if ( v3 != 5 )
      return 42;
    goto LABEL_23;
  }
  if ( v4 != 4 )
  {
LABEL_11:
    result = 33;
    if ( v4 == 20 )
      return result;
    return 42;
  }
  result = 33;
  if ( *(_QWORD *)(a2 - 64) == *(_QWORD *)(a1 - 64) )
    return 42;
  return result;
}
