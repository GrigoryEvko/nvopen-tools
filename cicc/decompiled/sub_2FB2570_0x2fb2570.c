// Function: sub_2FB2570
// Address: 0x2fb2570
//
__int64 __fastcall sub_2FB2570(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r8
  int v5; // ecx
  __int64 v6; // rdi
  unsigned int v7; // r9d
  __int64 *v8; // rax
  __int64 v9; // r11
  _QWORD **v10; // r15
  __int64 v11; // r9
  __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // r14
  unsigned int v16; // r13d
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // r11
  _QWORD **v20; // r9
  _QWORD *v21; // rax
  unsigned int i; // edx
  __int64 v23; // r10
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned int v26; // eax
  __int64 *v27; // rax
  unsigned int v29; // edx
  __int64 *v30; // rdx
  __int64 *v31; // rcx
  int v32; // eax
  int v33; // r10d
  int v34; // eax
  int v35; // ebx
  __int64 v36; // [rsp+0h] [rbp-40h]
  __int64 *v37; // [rsp+8h] [rbp-38h]

  if ( a2 == a3 )
    return a2;
  v4 = *(_QWORD *)(*a1 + 24LL);
  v5 = *(_DWORD *)(v4 + 24);
  v6 = *(_QWORD *)(v4 + 8);
  if ( !v5 )
  {
LABEL_51:
    v10 = 0;
    goto LABEL_5;
  }
  v7 = (v5 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a3 != *v8 )
  {
    v34 = 1;
    while ( v9 != -4096 )
    {
      v35 = v34 + 1;
      v7 = (v5 - 1) & (v34 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a3 == *v8 )
        goto LABEL_4;
      v34 = v35;
    }
    goto LABEL_51;
  }
LABEL_4:
  v10 = (_QWORD **)v8[1];
LABEL_5:
  v11 = a1[4];
  if ( a3 )
  {
    v12 = (unsigned int)(*(_DWORD *)(a3 + 24) + 1);
    v13 = v12;
  }
  else
  {
    v12 = 0;
    v13 = 0;
  }
  v14 = 0;
  if ( *(_DWORD *)(v11 + 32) > v13 )
    v14 = *(_QWORD *)(*(_QWORD *)(v11 + 24) + 8 * v12);
  v15 = a2;
  if ( !v5 )
    return v15;
  v16 = -1;
  while ( 1 )
  {
    v17 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v18 = (__int64 *)(v6 + 16LL * v17);
    v19 = *v18;
    if ( a2 != *v18 )
      break;
LABEL_12:
    v20 = (_QWORD **)v18[1];
    if ( !v20 || v10 == v20 )
      return a2;
    v21 = *v20;
    for ( i = 1; v21; ++i )
      v21 = (_QWORD *)*v21;
    if ( v16 > i )
    {
      v16 = i;
      v15 = a2;
    }
    v23 = a1[4];
    v24 = *v20[4];
    if ( v24 )
    {
      v25 = (unsigned int)(*(_DWORD *)(v24 + 24) + 1);
      v26 = *(_DWORD *)(v24 + 24) + 1;
    }
    else
    {
      v25 = 0;
      v26 = 0;
    }
    if ( v26 >= *(_DWORD *)(v23 + 32) )
      BUG();
    v27 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v23 + 24) + 8 * v25) + 8LL);
    if ( !v27 )
      return v15;
    if ( (__int64 *)v14 == v27 )
      goto LABEL_30;
    if ( !v14 )
      return v15;
    if ( v14 == v27[1] )
    {
LABEL_30:
      a2 = *v27;
      if ( !v5 )
        return a2;
    }
    else
    {
      if ( v27 == *(__int64 **)(v14 + 8) || *(_DWORD *)(v14 + 16) >= *((_DWORD *)v27 + 4) )
        return v15;
      if ( *(_BYTE *)(v23 + 112) )
      {
        if ( *((_DWORD *)v27 + 18) < *(_DWORD *)(v14 + 72) || *((_DWORD *)v27 + 19) > *(_DWORD *)(v14 + 76) )
          return v15;
        goto LABEL_30;
      }
      v29 = *(_DWORD *)(v23 + 116) + 1;
      *(_DWORD *)(v23 + 116) = v29;
      if ( v29 > 0x20 )
      {
        v37 = v27;
        v36 = v4;
        sub_2E6D080(v23);
        v27 = v37;
        if ( *((_DWORD *)v37 + 18) < *(_DWORD *)(v14 + 72) || *((_DWORD *)v37 + 19) > *(_DWORD *)(v14 + 76) )
          return v15;
        v4 = v36;
        v6 = *(_QWORD *)(v36 + 8);
        v5 = *(_DWORD *)(v36 + 24);
        goto LABEL_30;
      }
      v30 = v27;
      do
      {
        v31 = v30;
        v30 = (__int64 *)v30[1];
      }
      while ( v30 && *(_DWORD *)(v14 + 16) <= *((_DWORD *)v30 + 4) );
      if ( (__int64 *)v14 != v31 )
        return v15;
      v5 = *(_DWORD *)(v4 + 24);
      v6 = *(_QWORD *)(v4 + 8);
      a2 = *v27;
      if ( !v5 )
        return a2;
    }
  }
  v32 = 1;
  while ( v19 != -4096 )
  {
    v33 = v32 + 1;
    v17 = (v5 - 1) & (v32 + v17);
    v18 = (__int64 *)(v6 + 16LL * v17);
    v19 = *v18;
    if ( a2 == *v18 )
      goto LABEL_12;
    v32 = v33;
  }
  return a2;
}
