// Function: sub_F509B0
// Address: 0xf509b0
//
char __fastcall sub_F509B0(unsigned __int8 *a1, __int64 *a2)
{
  int v2; // eax
  int v3; // ebx
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  unsigned __int8 *v7; // rax
  unsigned __int8 *v8; // rbx
  char result; // al
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // edx
  __int64 v14; // rax
  unsigned int v15; // ebx
  __int64 v16; // r13
  int v17; // ebx
  unsigned __int8 *v18; // rdx
  int v19; // ecx
  __int64 i; // rax
  __int64 v21; // rdx
  unsigned int v22; // eax
  unsigned __int8 *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rax
  unsigned int v26; // ebx
  bool v27; // al

  v2 = *a1;
  if ( (unsigned int)(v2 - 30) <= 0xA )
    return 0;
  v3 = *a1;
  v4 = (unsigned int)(v2 - 39);
  if ( (unsigned int)v4 <= 0x38 )
  {
    v5 = 0x100060000000001LL;
    if ( _bittest64(&v5, v4) )
      return 0;
  }
  if ( (_BYTE)v3 == 85 )
  {
    v10 = *((_QWORD *)a1 - 4);
    if ( v10 )
    {
      if ( !*(_BYTE *)v10 && *(_QWORD *)(v10 + 24) == *((_QWORD *)a1 + 10) && (*(_BYTE *)(v10 + 33) & 0x20) != 0 )
      {
        v22 = *(_DWORD *)(v10 + 36);
        if ( v22 > 0x45 )
        {
          if ( v22 == 71 )
            return 0;
        }
        else if ( v22 > 0x43 )
        {
          return 0;
        }
      }
      if ( !*(_BYTE *)v10
        && *(_QWORD *)(v10 + 24) == *((_QWORD *)a1 + 10)
        && (*(_BYTE *)(v10 + 33) & 0x20) != 0
        && *(_DWORD *)(v10 + 36) == 70 )
      {
        return *(_QWORD *)(*(_QWORD *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)] + 24LL) == 0;
      }
    }
  }
  else if ( (unsigned __int8)(v3 - 34) > 0x33u || (v11 = 0x8000000000041LL, !_bittest64(&v11, (unsigned int)(v3 - 34))) )
  {
    if ( !(unsigned __int8)sub_B46900(a1) )
      return 0;
    if ( !(unsigned __int8)sub_B46970(a1) )
      return 1;
    goto LABEL_8;
  }
  if ( (unsigned __int8)sub_D5CD30(a1, a2) )
    return 1;
  if ( (unsigned __int8)sub_B46900(a1) )
  {
    if ( !(unsigned __int8)sub_B46970(a1) )
      return 1;
    v3 = *a1;
    if ( (_BYTE)v3 != 85 )
      goto LABEL_8;
    v16 = *((_QWORD *)a1 - 4);
    if ( !v16 || *(_BYTE *)v16 || *(_QWORD *)(v16 + 24) != *((_QWORD *)a1 + 10) || (*(_BYTE *)(v16 + 33) & 0x20) == 0 )
      goto LABEL_10;
    v17 = *(_DWORD *)(v16 + 36);
    if ( v17 == 208 || v17 == 343 || (unsigned int)(v17 - 5) <= 1 )
      return 1;
    if ( sub_B46A10((__int64)a1) )
    {
      v18 = *(unsigned __int8 **)&a1[32 * (1LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))];
      v19 = *v18;
      if ( (unsigned int)(v19 - 12) <= 1 )
        return 1;
      if ( (unsigned __int8)v19 <= 0x3Cu && ((0x100000000040000FuLL >> v19) & 1) != 0 )
      {
        for ( i = *((_QWORD *)v18 + 2); i; i = *(_QWORD *)(i + 8) )
        {
          v21 = *(_QWORD *)(i + 24);
          if ( *(_BYTE *)v21 != 85 )
            break;
          v24 = *(_QWORD *)(v21 - 32);
          if ( !v24 )
            break;
          if ( *(_BYTE *)v24 )
            break;
          if ( *(_QWORD *)(v24 + 24) != *(_QWORD *)(v21 + 80) )
            break;
          if ( (*(_BYTE *)(v24 + 33) & 0x20) == 0 )
            break;
          if ( (unsigned int)(*(_DWORD *)(v24 + 36) - 210) > 1 )
            break;
        }
        return i == 0;
      }
      return 0;
    }
    if ( v17 != 11 )
    {
LABEL_64:
      if ( !*(_BYTE *)v16 && *(_QWORD *)(v16 + 24) == *((_QWORD *)a1 + 10) && (*(_BYTE *)(v16 + 33) & 0x20) != 0 )
      {
        if ( (unsigned __int8)sub_B5A1B0((__int64)a1) )
          return (unsigned __int8)sub_B59EF0(a1, (__int64)a2) != 2;
        v3 = *a1;
        goto LABEL_8;
      }
LABEL_10:
      v7 = (unsigned __int8 *)sub_D5D560((__int64)a1, a2);
      v8 = v7;
      if ( v7 && *v7 <= 0x15u )
      {
        if ( !sub_AC30F0((__int64)v7) )
          return (unsigned int)*v8 - 12 <= 1;
        return 1;
      }
      if ( (unsigned __int8)sub_9755E0(a1, a2) )
        return 1;
      LOBYTE(v3) = *a1;
LABEL_71:
      if ( (_BYTE)v3 == 61 )
      {
        v23 = sub_BD3990(*((unsigned __int8 **)a1 - 4), (__int64)a2);
        if ( *v23 == 3 && (a1[2] & 1) == 0 )
          return v23[80] & 1;
      }
      return 0;
    }
    if ( sub_CF91F0((__int64)a1) )
    {
      v25 = *(_QWORD *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      if ( *(_BYTE *)v25 == 17 )
      {
        v26 = *(_DWORD *)(v25 + 32);
        if ( v26 <= 0x40 )
          v27 = *(_QWORD *)(v25 + 24) == 0;
        else
          v27 = v26 == (unsigned int)sub_C444A0(v25 + 24);
        return !v27;
      }
      return 0;
    }
    v3 = *a1;
    if ( (_BYTE)v3 == 85 )
    {
      v16 = *((_QWORD *)a1 - 4);
      if ( !v16 )
        goto LABEL_10;
      goto LABEL_64;
    }
LABEL_8:
    if ( (unsigned __int8)(v3 - 34) > 0x33u )
      return 0;
    v6 = 0x8000000000041LL;
    if ( !_bittest64(&v6, (unsigned int)(v3 - 34)) )
      goto LABEL_71;
    goto LABEL_10;
  }
  if ( *a1 != 85 )
    return 0;
  v12 = *((_QWORD *)a1 - 4);
  if ( !v12 || *(_BYTE *)v12 || *(_QWORD *)(v12 + 24) != *((_QWORD *)a1 + 10) || (*(_BYTE *)(v12 + 33) & 0x20) == 0 )
    return 0;
  v13 = *(_DWORD *)(v12 + 36);
  if ( v13 == 293 )
    return 1;
  if ( v13 <= 0x125 )
  {
    if ( v13 == 153 )
    {
      v14 = *(_QWORD *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      if ( *(_BYTE *)v14 == 17 )
      {
        v15 = *(_DWORD *)(v14 + 32);
        if ( v15 <= 0x40 )
        {
          if ( *(_QWORD *)(v14 + 24) == 1 )
            return 1;
        }
        else if ( (unsigned int)sub_C444A0(v14 + 24) == v15 - 1 )
        {
          return 1;
        }
      }
    }
    return 0;
  }
  result = 1;
  if ( v13 != 295 && v13 - 14255 > 1 )
    return 0;
  return result;
}
