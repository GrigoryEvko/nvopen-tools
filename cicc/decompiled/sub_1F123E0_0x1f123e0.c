// Function: sub_1F123E0
// Address: 0x1f123e0
//
__int64 __fastcall sub_1F123E0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  _QWORD *v11; // rax
  int v12; // r8d
  int v13; // r9d
  _QWORD *v14; // rcx
  _QWORD *v15; // rsi
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // ebx
  __int64 v20; // rax
  __int64 v21; // rbx
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  _QWORD *v24; // rax
  _QWORD *i; // rdx
  __int64 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 j; // rbx
  __int64 v33[5]; // [rsp+18h] [rbp-28h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 232) = a2;
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_41:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FCF76C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_41;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4FCF76C);
  v7 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 240) = v6;
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_43:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4FC6A0C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_43;
  }
  *(_QWORD *)(a1 + 248) = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(
                            *(_QWORD *)(v8 + 8),
                            &unk_4FC6A0C);
  v10 = *(unsigned int *)(*(_QWORD *)(a1 + 240) + 288LL);
  v11 = (_QWORD *)sub_2207820(112 * v10 + 8);
  v14 = v11;
  if ( v11 )
  {
    *v11 = v10;
    v15 = v11 + 1;
    if ( v10 )
    {
      v16 = v11 + 1;
      v14 += 14 * v10 + 1;
      do
      {
        v17 = v16 + 5;
        *v16 = 0;
        v16 += 14;
        *(v16 - 13) = 0;
        *(v16 - 11) = v17;
        *((_DWORD *)v16 - 20) = 0;
        *((_DWORD *)v16 - 19) = 4;
        *(v16 - 1) = 0;
      }
      while ( v14 != v16 );
    }
  }
  else
  {
    v15 = 0;
  }
  v18 = *(_QWORD *)(a1 + 240);
  *(_QWORD *)(a1 + 264) = v15;
  *(_DWORD *)(a1 + 472) = 0;
  v19 = *(_DWORD *)(v18 + 288);
  if ( v19 < *(_DWORD *)(a1 + 520) >> 2 || v19 > *(_DWORD *)(a1 + 520) )
  {
    _libc_free(*(_QWORD *)(a1 + 512));
    v20 = (__int64)_libc_calloc(v19, 1u);
    if ( !v20 )
    {
      if ( v19 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v20 = 0;
      }
      else
      {
        v20 = sub_13A3880(1u);
      }
    }
    *(_QWORD *)(a1 + 512) = v20;
    *(_DWORD *)(a1 + 520) = v19;
  }
  v21 = (__int64)(*(_QWORD *)(a2 + 104) - *(_QWORD *)(a2 + 96)) >> 3;
  v22 = *(unsigned int *)(a1 + 384);
  if ( (unsigned int)v21 >= v22 )
  {
    if ( (unsigned int)v21 <= v22 )
      goto LABEL_26;
    if ( (unsigned int)v21 > (unsigned __int64)*(unsigned int *)(a1 + 388) )
    {
      sub_16CD150(
        a1 + 376,
        (const void *)(a1 + 392),
        (unsigned int)((__int64)(*(_QWORD *)(a2 + 104) - *(_QWORD *)(a2 + 96)) >> 3),
        8,
        v12,
        v13);
      v22 = *(unsigned int *)(a1 + 384);
    }
    v23 = *(_QWORD *)(a1 + 376);
    v24 = (_QWORD *)(v23 + 8 * v22);
    for ( i = (_QWORD *)(v23 + 8LL * (unsigned int)v21); i != v24; ++v24 )
    {
      if ( v24 )
        *v24 = 0;
    }
  }
  *(_DWORD *)(a1 + 384) = v21;
LABEL_26:
  v26 = *(__int64 **)(a1 + 8);
  v27 = *v26;
  v28 = v26[1];
  if ( v27 == v28 )
LABEL_42:
    BUG();
  while ( *(_UNKNOWN **)v27 != &unk_4FC453D )
  {
    v27 += 16;
    if ( v28 == v27 )
      goto LABEL_42;
  }
  v29 = a2 + 320;
  v30 = (*(__int64 (__fastcall **)(_QWORD, void *, __int64, _QWORD *))(**(_QWORD **)(v27 + 8) + 104LL))(
          *(_QWORD *)(v27 + 8),
          &unk_4FC453D,
          v28,
          v14);
  *(_QWORD *)(a1 + 256) = v30;
  v33[0] = sub_1DDC5F0(v30);
  sub_1F123B0(a1, v33);
  for ( j = *(_QWORD *)(v29 + 8); v29 != j; j = *(_QWORD *)(j + 8) )
    *(_QWORD *)(*(_QWORD *)(a1 + 376) + 8LL * *(unsigned int *)(j + 48)) = sub_1DDC3C0(*(_QWORD *)(a1 + 256), j);
  return 0;
}
