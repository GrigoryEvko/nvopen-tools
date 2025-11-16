// Function: sub_2AB4060
// Address: 0x2ab4060
//
bool __fastcall sub_2AB4060(__int64 *a1)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 *v10; // r12
  __int64 *i; // r13
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 j; // r15
  __int64 v20; // rsi
  _QWORD *v21; // rax
  _QWORD *v22; // rdx
  __int64 k; // r15
  __int64 v24; // rsi
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // r12

  v2 = sub_AA5930(**(_QWORD **)(*a1 + 32));
  if ( v2 != v3 )
  {
    v4 = v3;
    v5 = v2;
    while ( !(unsigned __int8)sub_31A6BC0(a1[5], v5) )
    {
      if ( !v5 )
        BUG();
      v6 = *(_QWORD *)(v5 + 32);
      if ( !v6 )
        BUG();
      v5 = 0;
      if ( *(_BYTE *)(v6 - 24) == 84 )
        v5 = v6 - 24;
      if ( v4 == v5 )
        goto LABEL_11;
    }
    if ( v4 != v5 )
      return 0;
  }
LABEL_11:
  v8 = a1[5];
  v9 = *a1;
  v10 = *(__int64 **)(v8 + 160);
  for ( i = &v10[11 * *(unsigned int *)(v8 + 168)]; i != v10; v10 += 11 )
  {
    v12 = *v10;
    v13 = sub_D47930(v9);
    v14 = *(_QWORD *)(v12 - 8);
    v15 = v13;
    if ( (*(_DWORD *)(v12 + 4) & 0x7FFFFFF) != 0 )
    {
      v16 = 0;
      while ( v15 != *(_QWORD *)(v14 + 32LL * *(unsigned int *)(v12 + 72) + 8 * v16) )
      {
        if ( (*(_DWORD *)(v12 + 4) & 0x7FFFFFF) == (_DWORD)++v16 )
          goto LABEL_39;
      }
      v17 = 32 * v16;
    }
    else
    {
LABEL_39:
      v17 = 0x1FFFFFFFE0LL;
    }
    v18 = *(_QWORD *)(v14 + v17);
    v9 = *a1;
    for ( j = *(_QWORD *)(v18 + 16); j; v9 = *a1 )
    {
      while ( 1 )
      {
        v20 = *(_QWORD *)(*(_QWORD *)(j + 24) + 40LL);
        if ( !*(_BYTE *)(v9 + 84) )
          break;
        v21 = *(_QWORD **)(v9 + 64);
        v22 = &v21[*(unsigned int *)(v9 + 76)];
        if ( v21 == v22 )
          return 0;
        while ( v20 != *v21 )
        {
          if ( v22 == ++v21 )
            return 0;
        }
        j = *(_QWORD *)(j + 8);
        if ( !j )
          goto LABEL_24;
      }
      if ( !sub_C8CA60(v9 + 56, v20) )
        return 0;
      j = *(_QWORD *)(j + 8);
    }
LABEL_24:
    for ( k = *(_QWORD *)(*v10 + 16); k; v9 = *a1 )
    {
      while ( 1 )
      {
        v24 = *(_QWORD *)(*(_QWORD *)(k + 24) + 40LL);
        if ( !*(_BYTE *)(v9 + 84) )
          break;
        v25 = *(_QWORD **)(v9 + 64);
        v26 = &v25[*(unsigned int *)(v9 + 76)];
        if ( v25 == v26 )
          return 0;
        while ( v24 != *v25 )
        {
          if ( v26 == ++v25 )
            return 0;
        }
        k = *(_QWORD *)(k + 8);
        if ( !k )
          goto LABEL_31;
      }
      if ( !sub_C8CA60(v9 + 56, v24) )
        return 0;
      k = *(_QWORD *)(k + 8);
    }
LABEL_31:
    ;
  }
  v27 = sub_D46F00(v9);
  return v27 == sub_D47930(*a1);
}
