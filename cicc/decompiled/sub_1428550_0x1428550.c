// Function: sub_1428550
// Address: 0x1428550
//
char __fastcall sub_1428550(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  _QWORD *v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // r15
  _QWORD *v10; // r14
  __int64 v11; // rax
  int v12; // edx
  int v13; // eax
  __int64 v14; // rcx
  unsigned int v15; // esi
  __int64 *v16; // rdx
  __int64 v17; // rdi
  unsigned __int64 v18; // r8
  unsigned int v19; // esi
  __int64 *v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rcx
  int v24; // edx
  int v25; // r9d
  int v26; // edx
  _QWORD *v27; // rdx
  int v28; // r8d
  __int64 v29; // r8
  char result; // al

  if ( a2 == a3 )
    return 1;
  if ( a3 == *(_QWORD *)(a1 + 120) )
    return 0;
  v29 = *(_QWORD *)(a2 + 64);
  if ( v29 != *(_QWORD *)(a3 + 64) )
    return sub_15CC8F0(*(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 64), *(_QWORD *)(a3 + 64), a4, v29);
  if ( a2 == a3 )
    return 1;
  v5 = *(_QWORD *)(a1 + 120);
  if ( a3 == v5 )
    return 0;
  result = 1;
  if ( a2 == v5 )
    return result;
  v7 = *(_QWORD **)(a1 + 144);
  v8 = *(_QWORD **)(a1 + 136);
  v9 = *(_QWORD *)(a2 + 64);
  if ( v7 == v8 )
  {
    v10 = &v8[*(unsigned int *)(a1 + 156)];
    if ( v8 == v10 )
    {
      v27 = *(_QWORD **)(a1 + 136);
    }
    else
    {
      do
      {
        if ( v9 == *v8 )
          break;
        ++v8;
      }
      while ( v10 != v8 );
      v27 = v10;
    }
  }
  else
  {
    v10 = &v7[*(unsigned int *)(a1 + 152)];
    v8 = (_QWORD *)sub_16CC9F0(a1 + 128, *(_QWORD *)(a2 + 64));
    if ( v9 == *v8 )
    {
      v22 = *(_QWORD *)(a1 + 144);
      if ( v22 == *(_QWORD *)(a1 + 136) )
        v23 = *(unsigned int *)(a1 + 156);
      else
        v23 = *(unsigned int *)(a1 + 152);
      v27 = (_QWORD *)(v22 + 8 * v23);
    }
    else
    {
      v11 = *(_QWORD *)(a1 + 144);
      if ( v11 != *(_QWORD *)(a1 + 136) )
      {
        v8 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(a1 + 152));
        goto LABEL_15;
      }
      v8 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(a1 + 156));
      v27 = v8;
    }
  }
  if ( v8 != v27 )
  {
    while ( *v8 >= 0xFFFFFFFFFFFFFFFELL )
    {
      if ( v27 == ++v8 )
      {
        if ( v8 != v10 )
          goto LABEL_16;
        goto LABEL_31;
      }
    }
  }
LABEL_15:
  if ( v8 == v10 )
LABEL_31:
    sub_1427EB0(a1, v9);
LABEL_16:
  v12 = *(_DWORD *)(a1 + 320);
  result = 0;
  if ( v12 )
  {
    v13 = v12 - 1;
    v14 = *(_QWORD *)(a1 + 304);
    v15 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v16 = (__int64 *)(v14 + 16LL * ((v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
    v17 = *v16;
    if ( a2 == *v16 )
    {
LABEL_18:
      v18 = v16[1];
    }
    else
    {
      v26 = 1;
      while ( v17 != -8 )
      {
        v28 = v26 + 1;
        v15 = v13 & (v26 + v15);
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( a2 == *v16 )
          goto LABEL_18;
        v26 = v28;
      }
      v18 = 0;
    }
    v19 = v13 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v20 = (__int64 *)(v14 + 16LL * v19);
    v21 = *v20;
    if ( a3 == *v20 )
      return v20[1] > v18;
    v24 = 1;
    while ( v21 != -8 )
    {
      v25 = v24 + 1;
      v19 = v13 & (v24 + v19);
      v20 = (__int64 *)(v14 + 16LL * v19);
      v21 = *v20;
      if ( a3 == *v20 )
        return v20[1] > v18;
      v24 = v25;
    }
    return 0;
  }
  return result;
}
