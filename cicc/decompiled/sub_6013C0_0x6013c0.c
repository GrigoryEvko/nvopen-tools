// Function: sub_6013C0
// Address: 0x6013c0
//
__int64 __fastcall sub_6013C0(__int64 a1)
{
  __int64 v1; // r13
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rbx
  __int64 v11; // rdi
  __int64 i; // rbx
  __int64 v13; // rdi
  __int64 v14; // rbx
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rbx
  __int64 v18; // rcx
  int v19; // r14d
  char v20; // al
  __int64 v21; // rax
  bool v22; // zf
  _QWORD v23[7]; // [rsp+8h] [rbp-38h] BYREF

  v1 = *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  result = *(unsigned __int8 *)(v1 + 183);
  if ( (result & 6) != 0 )
    return result;
  v3 = *(_QWORD *)(v1 + 24);
  if ( v3 )
  {
    if ( (*(_BYTE *)(v1 + 177) & 2) == 0 )
    {
      v5 = *(_QWORD *)(v3 + 88);
      if ( (*(_BYTE *)(v5 + 193) & 2) == 0 && (*(_BYTE *)(v5 + 206) & 0x10) == 0 )
      {
        result = (unsigned int)result | 4;
        *(_BYTE *)(v1 + 183) = result;
        return result;
      }
    }
  }
  result = *(unsigned __int8 *)(a1 + 176);
  if ( (result & 4) != 0 )
  {
    if ( *(_BYTE *)(a1 + 140) != 11 )
    {
LABEL_8:
      *(_BYTE *)(v1 + 183) |= 4u;
      return result;
    }
    if ( (result & 0x10) == 0 )
    {
      v6 = *(_QWORD *)(a1 + 160);
LABEL_19:
      if ( !dword_4F077B4 )
      {
        if ( !dword_4F077BC )
          goto LABEL_21;
        goto LABEL_15;
      }
LABEL_28:
      for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 8LL); i; i = *(_QWORD *)(i + 8) )
      {
        v13 = *(_QWORD *)(i + 40);
        if ( (*(_BYTE *)(v13 + 141) & 0x20) == 0 && (*(_BYTE *)(v13 + 177) & 0x20) == 0 )
        {
          result = sub_8D4160(v13);
          if ( !(_DWORD)result )
            goto LABEL_8;
        }
      }
      v7 = 7;
      result = sub_72FD90(v6, 7);
      v14 = result;
      if ( result )
      {
        while ( 1 )
        {
          result = sub_8D4290(*(_QWORD *)(v14 + 120));
          if ( !(_DWORD)result )
            goto LABEL_8;
          v15 = *(_QWORD *)(v14 + 120);
          if ( (*(_BYTE *)(v15 + 140) & 0xFB) == 8 )
          {
            result = sub_8D4C10(v15, dword_4F077C4 != 2);
            if ( (result & 2) != 0 )
              goto LABEL_8;
          }
          v7 = 7;
          result = sub_72FD90(*(_QWORD *)(v14 + 112), 7);
          v14 = result;
          if ( !result )
            goto LABEL_42;
        }
      }
      goto LABEL_42;
    }
  }
  else if ( (result & 0x10) == 0 )
  {
    v6 = *(_QWORD *)(a1 + 160);
    if ( *(_BYTE *)(a1 + 140) != 11 )
      goto LABEL_28;
    goto LABEL_19;
  }
  result = (__int64)&dword_4F077BC;
  if ( !dword_4F077BC )
    goto LABEL_8;
  result = (__int64)&dword_4F077B4;
  if ( dword_4F077B4 )
    goto LABEL_8;
  result = (__int64)&qword_4F077A8;
  if ( !qword_4F077A8 )
    goto LABEL_8;
  v6 = *(_QWORD *)(a1 + 160);
  if ( *(_BYTE *)(a1 + 140) != 11 )
    goto LABEL_28;
LABEL_15:
  if ( qword_4F077A8 <= 0x1869Fu )
    goto LABEL_28;
LABEL_21:
  v7 = 7;
  result = sub_72FD90(v6, 7);
  v10 = result;
  if ( result )
  {
    while ( 1 )
    {
      if ( (unsigned int)sub_8D4290(*(_QWORD *)(v10 + 120)) )
      {
        v11 = *(_QWORD *)(v10 + 120);
        result = *(_BYTE *)(v11 + 140) & 0xFB;
        if ( (*(_BYTE *)(v11 + 140) & 0xFB) != 8 )
          break;
        v7 = dword_4F077C4 != 2;
        result = sub_8D4C10(v11, v7);
        if ( (result & 2) == 0 )
          break;
      }
      v7 = 7;
      result = sub_72FD90(*(_QWORD *)(v10 + 112), 7);
      v10 = result;
      if ( !result )
        goto LABEL_8;
    }
  }
LABEL_42:
  v16 = *(_QWORD *)a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 80LL) == 4 )
  {
    result = sub_87A660();
    if ( (_DWORD)result )
    {
      result = (__int64)&dword_4D041E0;
      if ( !dword_4D041E0 )
      {
        result = (__int64)&dword_4F077BC;
        if ( !dword_4F077BC )
          goto LABEL_8;
        result = (__int64)&dword_4F077B4;
        if ( dword_4F077B4 )
          goto LABEL_8;
        result = (__int64)&qword_4F077A8;
        if ( qword_4F077A8 > 0x1387Fu )
          goto LABEL_8;
      }
LABEL_63:
      *(_BYTE *)(v1 + 183) |= 2u;
      return result;
    }
  }
  if ( (*(_BYTE *)(v1 + 178) & 0x40) != 0 )
    goto LABEL_63;
  result = *(_QWORD *)(v1 + 16);
  if ( result )
  {
    result = *(_QWORD *)(result + 88);
    if ( (*(_BYTE *)(result + 193) & 2) != 0 )
      goto LABEL_63;
  }
  v17 = *(_QWORD *)(v1 + 8);
  if ( !v17 && (*(_BYTE *)(a1 + 179) & 4) == 0 )
    goto LABEL_63;
  result = *(_QWORD *)(a1 + 168);
  if ( *(_BYTE *)(result + 113) )
    goto LABEL_63;
  result = (__int64)&word_4D04898;
  v18 = word_4D04898;
  if ( !word_4D04898 )
    goto LABEL_8;
  result = *(unsigned __int8 *)(v1 + 183);
  if ( v17 && (result & 8) != 0 )
  {
    v19 = 0;
    if ( *(_BYTE *)(v17 + 80) != 17 )
      goto LABEL_52;
    v17 = *(_QWORD *)(v17 + 88);
    if ( v17 )
    {
      v19 = 1;
      while ( 1 )
      {
LABEL_52:
        v20 = *(_BYTE *)(v17 + 80);
        if ( v20 == 10 )
        {
          v21 = *(_QWORD *)(v17 + 88);
          if ( (*(_BYTE *)(v21 + 193) & 2) != 0 )
          {
            v16 = *(_QWORD *)(v21 + 152);
            v7 = a1;
            if ( !(unsigned int)sub_72F3C0(v16, a1, v23, 1, 1) )
            {
LABEL_73:
              result = *(unsigned __int8 *)(v1 + 183) | 2u;
              *(_BYTE *)(v1 + 183) |= 2u;
              break;
            }
          }
        }
        else if ( v20 == 20 && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v17 + 88) + 176LL) + 193LL) & 2) != 0 )
        {
          goto LABEL_73;
        }
        if ( v19 )
        {
          v17 = *(_QWORD *)(v17 + 8);
          if ( v17 )
            continue;
        }
        result = *(unsigned __int8 *)(v1 + 183);
        break;
      }
    }
  }
  if ( (result & 0x16) == 0 )
  {
    v16 = a1;
    v22 = (unsigned int)sub_600530(a1) == 0;
    result = *(unsigned __int8 *)(v1 + 183);
    if ( !v22 )
    {
      result = (unsigned int)result | 2;
      *(_BYTE *)(v1 + 183) = result;
    }
  }
  if ( !dword_4D04964 && (result & 6) == 0 )
  {
    v23[0] = sub_724DC0(v16, v7, (unsigned int)dword_4D04964, v18, v8, v9);
    if ( (*(_BYTE *)(v1 + 176) & 4) == 0 )
    {
      if ( (unsigned int)sub_72FDF0(a1, v23[0]) )
        *(_BYTE *)(v1 + 183) |= 2u;
    }
    sub_724E30(v23);
    result = *(unsigned __int8 *)(v1 + 183);
  }
  if ( (result & 2) == 0 )
    goto LABEL_8;
  return result;
}
