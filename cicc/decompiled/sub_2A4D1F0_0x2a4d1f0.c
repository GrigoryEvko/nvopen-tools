// Function: sub_2A4D1F0
// Address: 0x2a4d1f0
//
_QWORD **__fastcall sub_2A4D1F0(__int64 a1, __int64 a2)
{
  char v2; // dl
  _QWORD **v4; // rax
  __int64 v5; // rcx
  _QWORD **v6; // r13
  _QWORD *v7; // rdi
  _QWORD **v8; // r14
  __int64 v9; // r12
  char v10; // dl
  _QWORD **result; // rax
  __int64 v12; // rcx
  _QWORD **v13; // r13
  _QWORD *v14; // rdi
  _QWORD **v15; // r14
  __int64 v16; // r12
  _QWORD **v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // rdx

  v2 = *(_BYTE *)(a1 + 2132);
  v4 = *(_QWORD ***)(a1 + 2112);
  if ( v2 )
    v5 = *(unsigned int *)(a1 + 2124);
  else
    v5 = *(unsigned int *)(a1 + 2120);
  v6 = &v4[v5];
  if ( v4 == v6 )
  {
LABEL_6:
    ++*(_QWORD *)(a1 + 2104);
    v9 = a1 + 2104;
    if ( v2 )
    {
LABEL_7:
      *(_QWORD *)(a1 + 2124) = 0;
      goto LABEL_8;
    }
  }
  else
  {
    while ( 1 )
    {
      v7 = *v4;
      v8 = v4;
      if ( (unsigned __int64)*v4 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v6 == ++v4 )
        goto LABEL_6;
    }
    v9 = a1 + 2104;
    if ( v6 != v4 )
    {
      do
      {
        sub_B43D60(v7);
        v17 = v8 + 1;
        if ( v8 + 1 == v6 )
          break;
        while ( 1 )
        {
          v7 = *v17;
          v8 = v17;
          if ( (unsigned __int64)*v17 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v6 == ++v17 )
            goto LABEL_19;
        }
      }
      while ( v6 != v17 );
LABEL_19:
      v2 = *(_BYTE *)(a1 + 2132);
    }
    ++*(_QWORD *)(a1 + 2104);
    if ( v2 )
      goto LABEL_7;
  }
  v18 = 4 * (*(_DWORD *)(a1 + 2124) - *(_DWORD *)(a1 + 2128));
  v19 = *(unsigned int *)(a1 + 2120);
  if ( v18 < 0x20 )
    v18 = 32;
  if ( v18 >= (unsigned int)v19 )
  {
    a2 = 0xFFFFFFFFLL;
    memset(*(void **)(a1 + 2112), -1, 8 * v19);
    goto LABEL_7;
  }
  sub_C8C990(v9, a2);
LABEL_8:
  v10 = *(_BYTE *)(a1 + 2228);
  result = *(_QWORD ***)(a1 + 2208);
  if ( v10 )
    v12 = *(unsigned int *)(a1 + 2220);
  else
    v12 = *(unsigned int *)(a1 + 2216);
  v13 = &result[v12];
  if ( result == v13 )
  {
LABEL_13:
    ++*(_QWORD *)(a1 + 2200);
    v16 = a1 + 2200;
    if ( v10 )
    {
LABEL_14:
      *(_QWORD *)(a1 + 2220) = 0;
      return result;
    }
  }
  else
  {
    while ( 1 )
    {
      v14 = *result;
      v15 = result;
      if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v13 == ++result )
        goto LABEL_13;
    }
    v16 = a1 + 2200;
    if ( v13 != result )
    {
      do
      {
        sub_B14290(v14);
        result = v15 + 1;
        if ( v15 + 1 == v13 )
          break;
        while ( 1 )
        {
          v14 = *result;
          v15 = result;
          if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v13 == ++result )
            goto LABEL_31;
        }
      }
      while ( v13 != result );
LABEL_31:
      v10 = *(_BYTE *)(a1 + 2228);
    }
    ++*(_QWORD *)(a1 + 2200);
    if ( v10 )
      goto LABEL_14;
  }
  v20 = 4 * (*(_DWORD *)(a1 + 2220) - *(_DWORD *)(a1 + 2224));
  v21 = *(unsigned int *)(a1 + 2216);
  if ( v20 < 0x20 )
    v20 = 32;
  if ( v20 >= (unsigned int)v21 )
  {
    result = (_QWORD **)memset(*(void **)(a1 + 2208), -1, 8 * v21);
    goto LABEL_14;
  }
  return (_QWORD **)sub_C8C990(v16, a2);
}
