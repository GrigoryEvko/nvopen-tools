// Function: sub_13CE210
// Address: 0x13ce210
//
__int64 __fastcall sub_13CE210(__int64 a1, _QWORD *a2, char a3)
{
  __int64 v3; // r14
  __int64 v4; // r13
  int v6; // r13d
  __int64 v7; // rax
  __int64 result; // rax
  unsigned int v9; // r15d
  unsigned int v10; // r15d
  unsigned int v11; // r15d
  bool v12; // al
  __int64 v13; // rax
  unsigned int v14; // r13d
  unsigned int v15; // r15d
  __int64 v16; // rax
  char v17; // si
  int v18; // [rsp+4h] [rbp-3Ch]
  int v19; // [rsp+8h] [rbp-38h]

  if ( !a1 )
    return 0;
  v3 = *(_QWORD *)(a1 - 48);
  if ( !v3 )
    return 0;
  v4 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v4 + 16) > 0x10u )
    return 0;
  if ( !(unsigned __int8)sub_1593BB0(*(_QWORD *)(a1 - 24)) )
  {
    if ( *(_BYTE *)(v4 + 16) == 13 )
    {
      v11 = *(_DWORD *)(v4 + 32);
      if ( v11 <= 0x40 )
        v12 = *(_QWORD *)(v4 + 24) == 0;
      else
        v12 = v11 == (unsigned int)sub_16A57B0(v4 + 24);
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v4 + 8LL) != 16 )
        return 0;
      v13 = sub_15A1020(v4);
      if ( !v13 || *(_BYTE *)(v13 + 16) != 13 )
      {
        v19 = *(_QWORD *)(*(_QWORD *)v4 + 32LL);
        if ( v19 )
        {
          v15 = 0;
          while ( 1 )
          {
            v16 = sub_15A0A60(v4, v15);
            if ( !v16 )
              return 0;
            v17 = *(_BYTE *)(v16 + 16);
            if ( v17 != 9 )
            {
              if ( v17 != 13 )
                return 0;
              if ( *(_DWORD *)(v16 + 32) <= 0x40u )
              {
                if ( *(_QWORD *)(v16 + 24) )
                  return 0;
              }
              else
              {
                v18 = *(_DWORD *)(v16 + 32);
                if ( v18 != (unsigned int)sub_16A57B0(v16 + 24) )
                  return 0;
              }
            }
            if ( v19 == ++v15 )
              goto LABEL_5;
          }
        }
        goto LABEL_5;
      }
      v14 = *(_DWORD *)(v13 + 32);
      if ( v14 <= 0x40 )
        v12 = *(_QWORD *)(v13 + 24) == 0;
      else
        v12 = v14 == (unsigned int)sub_16A57B0(v13 + 24);
    }
    if ( !v12 )
      return 0;
  }
LABEL_5:
  v6 = *(_WORD *)(a1 + 18) & 0x7FFF;
  if ( (unsigned int)(v6 - 32) > 1 )
    return 0;
  if ( !a2 )
    return 0;
  v7 = *(a2 - 6);
  if ( !v7 )
    return 0;
  if ( v3 == *(a2 - 3) )
  {
    v9 = *((_WORD *)a2 + 9) & 0x7FFF;
    if ( (unsigned __int8)sub_15FF7E0(v9) )
    {
      if ( v9 == 36 )
        goto LABEL_22;
      goto LABEL_14;
    }
    v7 = *(a2 - 6);
  }
  if ( v7 != v3 )
    return 0;
  if ( !*(a2 - 3) )
    return 0;
  v10 = *((_WORD *)a2 + 9) & 0x7FFF;
  if ( !(unsigned __int8)sub_15FF7E0(v10) )
    return 0;
  v9 = sub_15FF5D0(v10);
  if ( v9 == 36 )
  {
LABEL_22:
    if ( v6 == 33 )
    {
      result = a1;
      if ( a3 )
        return (__int64)a2;
    }
    else
    {
      result = 0;
      if ( a3 )
        return sub_15A0640(*a2);
    }
    return result;
  }
LABEL_14:
  result = 0;
  if ( v9 == 35 && !a3 )
  {
    result = (__int64)a2;
    if ( v6 == 33 )
      return sub_15A0600(*a2);
  }
  return result;
}
