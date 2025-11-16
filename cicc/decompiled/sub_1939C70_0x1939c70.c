// Function: sub_1939C70
// Address: 0x1939c70
//
__int64 __fastcall sub_1939C70(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v8; // eax
  char *v9; // r15
  _QWORD *v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // r8
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  char *v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rcx
  _QWORD *v21; // rdx
  __int64 *v22; // rsi
  unsigned int v23; // edi
  __int64 *v24; // rcx
  signed __int64 v25; // rax
  char *v26; // [rsp+0h] [rbp-40h]
  _QWORD *v27; // [rsp+8h] [rbp-38h]
  unsigned __int8 v28; // [rsp+8h] [rbp-38h]

  if ( *(_BYTE *)(a2 + 16) <= 0x17u )
    goto LABEL_3;
  LOBYTE(v8) = sub_15CCEE0(*a1, a2, a3);
  LODWORD(v9) = v8;
  if ( (_BYTE)v8 )
    goto LABEL_3;
  v11 = *(_QWORD **)(a4 + 16);
  v12 = *(_QWORD **)(a4 + 8);
  if ( v11 == v12 )
  {
    v13 = &v12[*(unsigned int *)(a4 + 28)];
    if ( v12 == v13 )
    {
      v21 = *(_QWORD **)(a4 + 8);
    }
    else
    {
      do
      {
        if ( a2 == *v12 )
          break;
        ++v12;
      }
      while ( v13 != v12 );
      v21 = v13;
    }
  }
  else
  {
    v27 = &v11[*(unsigned int *)(a4 + 24)];
    v12 = sub_16CC9F0(a4, a2);
    v13 = v27;
    if ( a2 == *v12 )
    {
      v19 = *(_QWORD *)(a4 + 16);
      if ( v19 == *(_QWORD *)(a4 + 8) )
        v20 = *(unsigned int *)(a4 + 28);
      else
        v20 = *(unsigned int *)(a4 + 24);
      v21 = (_QWORD *)(v19 + 8 * v20);
    }
    else
    {
      v14 = *(_QWORD *)(a4 + 16);
      if ( v14 != *(_QWORD *)(a4 + 8) )
      {
        v12 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a4 + 24));
        goto LABEL_9;
      }
      v12 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a4 + 28));
      v21 = v12;
    }
  }
  while ( v21 != v12 && *v12 >= 0xFFFFFFFFFFFFFFFELL )
    ++v12;
LABEL_9:
  if ( v12 != v13 )
  {
LABEL_3:
    LODWORD(v9) = 1;
    return (unsigned int)v9;
  }
  v28 = sub_14AF470(a2, a3, *a1, 0);
  if ( v28 && !(unsigned __int8)sub_15F2ED0(a2) )
  {
    v15 = *(__int64 **)(a4 + 8);
    if ( *(__int64 **)(a4 + 16) != v15 )
    {
LABEL_13:
      sub_16CCBA0(a4, a2);
      goto LABEL_14;
    }
    v22 = &v15[*(unsigned int *)(a4 + 28)];
    v23 = *(_DWORD *)(a4 + 28);
    if ( v15 == v22 )
    {
LABEL_60:
      if ( v23 >= *(_DWORD *)(a4 + 24) )
        goto LABEL_13;
      *(_DWORD *)(a4 + 28) = v23 + 1;
      *v22 = a2;
      ++*(_QWORD *)a4;
    }
    else
    {
      v24 = 0;
      while ( a2 != *v15 )
      {
        if ( *v15 == -2 )
          v24 = v15;
        if ( v22 == ++v15 )
        {
          if ( !v24 )
            goto LABEL_60;
          *v24 = a2;
          --*(_DWORD *)(a4 + 32);
          ++*(_QWORD *)a4;
          break;
        }
      }
    }
LABEL_14:
    v16 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    {
      v9 = *(char **)(a2 - 8);
      v26 = &v9[v16];
    }
    else
    {
      v26 = (char *)a2;
      v9 = (char *)(a2 - v16);
    }
    v17 = (__int64)(0xAAAAAAAAAAAAAAABLL * (v16 >> 3)) >> 2;
    if ( v17 )
    {
      v18 = &v9[96 * v17];
      while ( 1 )
      {
        if ( !(unsigned __int8)sub_1939C70(a1, *(_QWORD *)v9, a3, a4) )
        {
          LOBYTE(v9) = v26 == v9;
          return (unsigned int)v9;
        }
        if ( !(unsigned __int8)sub_1939C70(a1, *((_QWORD *)v9 + 3), a3, a4) )
        {
          LOBYTE(v9) = v26 == v9 + 24;
          return (unsigned int)v9;
        }
        if ( !(unsigned __int8)sub_1939C70(a1, *((_QWORD *)v9 + 6), a3, a4) )
        {
          LOBYTE(v9) = v26 == v9 + 48;
          return (unsigned int)v9;
        }
        if ( !(unsigned __int8)sub_1939C70(a1, *((_QWORD *)v9 + 9), a3, a4) )
          break;
        v9 += 96;
        if ( v9 == v18 )
          goto LABEL_47;
      }
      LOBYTE(v9) = v26 == v9 + 72;
      return (unsigned int)v9;
    }
    v18 = v9;
LABEL_47:
    v25 = v26 - v18;
    if ( v26 - v18 != 48 )
    {
      if ( v25 != 72 )
      {
        if ( v25 != 24 )
        {
          LODWORD(v9) = v28;
          return (unsigned int)v9;
        }
        goto LABEL_58;
      }
      if ( !(unsigned __int8)sub_1939C70(a1, *(_QWORD *)v18, a3, a4) )
      {
        LOBYTE(v9) = v26 == v18;
        return (unsigned int)v9;
      }
      v18 += 24;
    }
    if ( !(unsigned __int8)sub_1939C70(a1, *(_QWORD *)v18, a3, a4) )
      goto LABEL_59;
    v18 += 24;
LABEL_58:
    LODWORD(v9) = sub_1939C70(a1, *(_QWORD *)v18, a3, a4);
    if ( (_BYTE)v9 )
      return (unsigned int)v9;
LABEL_59:
    LOBYTE(v9) = v18 == v26;
  }
  return (unsigned int)v9;
}
