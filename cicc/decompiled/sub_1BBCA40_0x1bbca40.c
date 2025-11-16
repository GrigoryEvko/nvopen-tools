// Function: sub_1BBCA40
// Address: 0x1bbca40
//
_QWORD *__fastcall sub_1BBCA40(_QWORD *a1, _QWORD *a2, __int64 a3, int a4)
{
  _QWORD *result; // rax
  int v5; // r9d
  _QWORD *v6; // r10
  __int64 v7; // r8
  __int64 v8; // rdi
  _QWORD *v9; // rdx
  _QWORD *v10; // r8
  __int64 v11; // rdx
  __int64 v12; // rbx
  int v13; // edi
  char v14; // r8
  bool v15; // r15
  bool v16; // r13
  int v17; // edi
  int v18; // r14d
  __int64 v19; // r10
  __int64 v20; // r9
  int v21; // ecx
  char v22; // dl
  int v23; // r11d
  _QWORD **v24; // rcx
  _QWORD **v25; // r9
  __int64 v26; // rdx
  __int64 v27; // [rsp-8h] [rbp-8h]

  result = a1;
  v5 = a3;
  v6 = &a2[a3];
  v7 = (8 * a3) >> 5;
  v8 = (8 * a3) >> 3;
  if ( v7 > 0 )
  {
    v9 = a2;
    v10 = &a2[4 * v7];
    while ( *(_BYTE *)(*v9 + 16LL) > 0x17u )
    {
      if ( *(_BYTE *)(v9[1] + 16LL) <= 0x17u )
      {
        ++v9;
        goto LABEL_8;
      }
      if ( *(_BYTE *)(v9[2] + 16LL) <= 0x17u )
      {
        v9 += 2;
        goto LABEL_8;
      }
      if ( *(_BYTE *)(v9[3] + 16LL) <= 0x17u )
      {
        v9 += 3;
        goto LABEL_8;
      }
      v9 += 4;
      if ( v10 == v9 )
      {
        v8 = v6 - v9;
        goto LABEL_11;
      }
    }
    goto LABEL_8;
  }
  v9 = a2;
LABEL_11:
  switch ( v8 )
  {
    case 2LL:
LABEL_47:
      if ( *(_BYTE *)(*v9 + 16LL) <= 0x17u )
        goto LABEL_8;
      ++v9;
LABEL_14:
      if ( *(_BYTE *)(*v9 + 16LL) > 0x17u )
        break;
LABEL_8:
      if ( v6 != v9 )
      {
        result[1] = 0;
        v11 = a2[a4];
        result[2] = 0;
        *result = v11;
        return result;
      }
      break;
    case 3LL:
      if ( *(_BYTE *)(*v9 + 16LL) <= 0x17u )
        goto LABEL_8;
      ++v9;
      goto LABEL_47;
    case 1LL:
      goto LABEL_14;
  }
  v12 = a2[a4];
  v13 = *(unsigned __int8 *)(v12 + 16);
  v14 = *(_BYTE *)(v12 + 16);
  if ( (unsigned __int8)v13 > 0x17u )
  {
    v15 = (unsigned int)(v13 - 60) <= 0xC;
    v16 = (unsigned int)(v13 - 35) <= 0x11;
  }
  else
  {
    v15 = 0;
    v16 = 0;
  }
  v17 = v13 - 24;
  if ( v5 <= 0 )
  {
    v26 = a2[a4];
    goto LABEL_53;
  }
  *((_DWORD *)&v27 - 13) = a4;
  v18 = v17;
  v19 = 0;
  *(&v27 - 6) = (unsigned int)(v5 - 1);
  while ( 1 )
  {
    v20 = a2[v19];
    v21 = *(unsigned __int8 *)(v20 + 16);
    v22 = *(_BYTE *)(v20 + 16);
    v23 = v21 - 24;
    if ( v16 )
    {
      if ( (unsigned __int8)v21 <= 0x17u )
        goto LABEL_31;
      if ( (unsigned int)(v21 - 35) <= 0x11 )
        goto LABEL_22;
    }
    else if ( (unsigned __int8)v21 <= 0x17u )
    {
      goto LABEL_31;
    }
    if ( v15 && (unsigned int)(v21 - 60) <= 0xC )
    {
      if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
        v24 = *(_QWORD ***)(v12 - 8);
      else
        v24 = (_QWORD **)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
      if ( (*(_BYTE *)(v20 + 23) & 0x40) != 0 )
        v25 = *(_QWORD ***)(v20 - 8);
      else
        v25 = (_QWORD **)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
      if ( **v25 != **v24 )
      {
LABEL_33:
        *result = v12;
        result[1] = 0;
        result[2] = 0;
        return result;
      }
LABEL_22:
      if ( v22 != v14 && v23 != v18 )
      {
        if ( v17 != v18 )
          goto LABEL_33;
        *((_DWORD *)&v27 - 13) = v19;
        v18 = v23;
      }
      goto LABEL_26;
    }
LABEL_31:
    if ( v22 != v14 && v23 != v18 )
      goto LABEL_33;
LABEL_26:
    if ( *(&v27 - 6) == v19 )
      break;
    ++v19;
  }
  v26 = a2[*((unsigned int *)&v27 - 13)];
LABEL_53:
  *result = v12;
  result[1] = v12;
  result[2] = v26;
  return result;
}
