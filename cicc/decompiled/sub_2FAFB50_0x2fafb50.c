// Function: sub_2FAFB50
// Address: 0x2fafb50
//
__int64 __fastcall sub_2FAFB50(__int64 a1, unsigned int a2)
{
  __int64 v2; // r14
  __int64 v4; // r9
  __int64 v5; // r11
  __int64 v6; // r10
  _QWORD *v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rdi
  _QWORD *i; // r8
  bool v11; // cf
  int v12; // eax
  bool v13; // cl
  unsigned __int64 v14; // rax
  __int64 result; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // r8
  __int64 v18; // r14
  __int64 v19; // r13
  __int64 j; // r15
  __int64 v21; // rdx
  int v22; // ebx
  unsigned __int8 *v23; // rdx
  unsigned int v24; // ecx
  unsigned int v25; // eax
  _BYTE *v26; // r9
  __int64 v27; // rdi
  _DWORD *v28; // rdx
  __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-40h]

  v2 = 112LL * a2;
  v4 = *(_QWORD *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 216);
  v6 = v4 + v2;
  v7 = *(_QWORD **)(v4 + v2 + 24);
  v8 = *(_QWORD *)(v4 + v2);
  v9 = *(_QWORD *)(v4 + v2 + 8);
  for ( i = &v7[2 * *(unsigned int *)(v4 + v2 + 32)]; i != v7; v7 += 2 )
  {
    while ( 1 )
    {
      v12 = *(_DWORD *)(v4 + 112LL * *((unsigned int *)v7 + 2) + 16);
      if ( v12 != -1 )
        break;
      v11 = __CFADD__(*v7, v8);
      v8 += *v7;
      if ( v11 )
        v8 = -1;
      v7 += 2;
      if ( i == v7 )
        goto LABEL_11;
    }
    if ( v12 == 1 )
    {
      v11 = __CFADD__(*v7, v9);
      v9 += *v7;
      if ( v11 )
        v9 = -1;
    }
  }
LABEL_11:
  v13 = *(_DWORD *)(v6 + 16) > 0;
  v14 = v5 + v9;
  if ( __CFADD__(v5, v9) )
    v14 = -1;
  if ( v14 > v8 )
  {
    v11 = __CFADD__(v5, v8);
    v16 = v5 + v8;
    if ( v11 )
      v16 = -1;
    if ( v16 <= v9 )
    {
      *(_DWORD *)(v6 + 16) = 1;
      result = 0;
      if ( v13 )
        return result;
      goto LABEL_21;
    }
    *(_DWORD *)(v6 + 16) = 0;
  }
  else
  {
    *(_DWORD *)(v6 + 16) = -1;
  }
  result = 0;
  if ( !v13 )
    return result;
LABEL_21:
  v17 = *(_QWORD *)(a1 + 24);
  v18 = v17 + v2;
  v19 = *(_QWORD *)(v18 + 24);
  for ( j = v19 + 16LL * *(unsigned int *)(v18 + 32); j != v19; v19 += 16 )
  {
    v21 = *(unsigned int *)(v19 + 8);
    v22 = *(_DWORD *)(v19 + 8);
    if ( *(_DWORD *)(v18 + 16) != *(_DWORD *)(v17 + 112 * v21 + 16) )
    {
      v23 = (unsigned __int8 *)(*(_QWORD *)(a1 + 272) + v21);
      v24 = *(_DWORD *)(a1 + 232);
      v25 = *v23;
      v26 = v23;
      if ( v25 >= v24 )
        goto LABEL_31;
      v27 = *(_QWORD *)(a1 + 224);
      while ( 1 )
      {
        v28 = (_DWORD *)(v27 + 4LL * v25);
        if ( v22 == *v28 )
          break;
        v25 += 256;
        if ( v24 <= v25 )
          goto LABEL_31;
      }
      if ( v28 == (_DWORD *)(v27 + 4LL * v24) )
      {
LABEL_31:
        *v26 = v24;
        v29 = *(unsigned int *)(a1 + 232);
        if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 236) )
        {
          v30 = v17;
          sub_C8D5F0(a1 + 224, (const void *)(a1 + 240), v29 + 1, 4u, v17, (__int64)v26);
          v17 = v30;
          v29 = *(unsigned int *)(a1 + 232);
        }
        *(_DWORD *)(*(_QWORD *)(a1 + 224) + 4 * v29) = v22;
        ++*(_DWORD *)(a1 + 232);
      }
    }
  }
  return 1;
}
