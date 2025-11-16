// Function: sub_3063FF0
// Address: 0x3063ff0
//
_QWORD *__fastcall sub_3063FF0(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v4; // rdx
  int v5; // r10d
  _QWORD *v6; // rcx
  unsigned int v7; // r8d
  _QWORD *v8; // rax
  void *v9; // rdi
  __int64 *v10; // r12
  int v11; // r10d
  _QWORD *v12; // rcx
  unsigned int v13; // r8d
  _QWORD *result; // rax
  void *v15; // rdi
  __int64 *v16; // rbx
  int v17; // eax
  int v18; // edx
  _QWORD *v19; // rax
  __int64 v20; // rdi
  int v21; // eax
  int v22; // esi
  __int64 v23; // r8
  unsigned int v24; // eax
  int v25; // edx
  void *v26; // rdi
  int v27; // r10d
  _QWORD *v28; // r9
  int v29; // eax
  __int64 v30; // rdi
  int v31; // eax
  int v32; // esi
  __int64 v33; // r8
  unsigned int v34; // eax
  void *v35; // rdi
  int v36; // r10d
  _QWORD *v37; // r9
  int v38; // eax
  int v39; // eax
  __int64 v40; // rdi
  int v41; // r9d
  unsigned int v42; // r13d
  _QWORD *v43; // r8
  void *v44; // rsi
  int v45; // eax
  int v46; // eax
  __int64 v47; // rdi
  int v48; // r9d
  unsigned int v49; // r13d
  _QWORD *v50; // r8
  void *v51; // rsi

  v3 = *(_DWORD *)(a2 + 24);
  if ( v3 )
  {
    v4 = *(_QWORD *)(a2 + 8);
    v5 = 1;
    v6 = 0;
    v7 = (v3 - 1) & (((unsigned int)&unk_4F8D468 >> 9) ^ ((unsigned int)&unk_4F8D468 >> 4));
    v8 = (_QWORD *)(v4 + 16LL * v7);
    v9 = (void *)*v8;
    if ( (_UNKNOWN *)*v8 == &unk_4F8D468 )
    {
LABEL_3:
      v10 = v8 + 1;
      if ( v8[1] )
        goto LABEL_4;
      goto LABEL_20;
    }
    while ( v9 != (void *)-4096LL )
    {
      if ( v9 == (void *)-8192LL && !v6 )
        v6 = v8;
      v7 = (v3 - 1) & (v5 + v7);
      v8 = (_QWORD *)(v4 + 16LL * v7);
      v9 = (void *)*v8;
      if ( (_UNKNOWN *)*v8 == &unk_4F8D468 )
        goto LABEL_3;
      ++v5;
    }
    if ( !v6 )
      v6 = v8;
    v17 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v18 = v17 + 1;
    if ( 4 * (v17 + 1) < 3 * v3 )
    {
      if ( v3 - *(_DWORD *)(a2 + 20) - v18 <= v3 >> 3 )
      {
        sub_2275E10(a2, v3);
        v38 = *(_DWORD *)(a2 + 24);
        if ( !v38 )
          goto LABEL_91;
        v39 = v38 - 1;
        v40 = *(_QWORD *)(a2 + 8);
        v41 = 1;
        v42 = v39 & (((unsigned int)&unk_4F8D468 >> 9) ^ ((unsigned int)&unk_4F8D468 >> 4));
        v43 = 0;
        v18 = *(_DWORD *)(a2 + 16) + 1;
        v6 = (_QWORD *)(v40 + 16LL * v42);
        v44 = (void *)*v6;
        if ( (_UNKNOWN *)*v6 != &unk_4F8D468 )
        {
          while ( v44 != (void *)-4096LL )
          {
            if ( v44 == (void *)-8192LL && !v43 )
              v43 = v6;
            v42 = v39 & (v41 + v42);
            v6 = (_QWORD *)(v40 + 16LL * v42);
            v44 = (void *)*v6;
            if ( (_UNKNOWN *)*v6 == &unk_4F8D468 )
              goto LABEL_17;
            ++v41;
          }
          if ( v43 )
            v6 = v43;
        }
      }
      goto LABEL_17;
    }
  }
  else
  {
    ++*(_QWORD *)a2;
  }
  sub_2275E10(a2, 2 * v3);
  v31 = *(_DWORD *)(a2 + 24);
  if ( !v31 )
    goto LABEL_91;
  v32 = v31 - 1;
  v33 = *(_QWORD *)(a2 + 8);
  v34 = (v31 - 1) & (((unsigned int)&unk_4F8D468 >> 9) ^ ((unsigned int)&unk_4F8D468 >> 4));
  v18 = *(_DWORD *)(a2 + 16) + 1;
  v6 = (_QWORD *)(v33 + 16LL * v34);
  v35 = (void *)*v6;
  if ( (_UNKNOWN *)*v6 != &unk_4F8D468 )
  {
    v36 = 1;
    v37 = 0;
    while ( v35 != (void *)-4096LL )
    {
      if ( !v37 && v35 == (void *)-8192LL )
        v37 = v6;
      v34 = v32 & (v36 + v34);
      v6 = (_QWORD *)(v33 + 16LL * v34);
      v35 = (void *)*v6;
      if ( (_UNKNOWN *)*v6 == &unk_4F8D468 )
        goto LABEL_17;
      ++v36;
    }
    if ( v37 )
      v6 = v37;
  }
LABEL_17:
  *(_DWORD *)(a2 + 16) = v18;
  if ( *v6 != -4096 )
    --*(_DWORD *)(a2 + 20);
  v6[1] = 0;
  *v6 = &unk_4F8D468;
  v10 = v6 + 1;
LABEL_20:
  v19 = (_QWORD *)sub_22077B0(0x10u);
  if ( v19 )
    *v19 = &unk_4A0C138;
  v20 = *v10;
  *v10 = (__int64)v19;
  if ( v20 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
  v3 = *(_DWORD *)(a2 + 24);
  if ( !v3 )
  {
    ++*(_QWORD *)a2;
    goto LABEL_26;
  }
  v4 = *(_QWORD *)(a2 + 8);
LABEL_4:
  v11 = 1;
  v12 = 0;
  v13 = (v3 - 1) & (((unsigned int)&unk_5040920 >> 9) ^ ((unsigned int)&unk_5040920 >> 4));
  result = (_QWORD *)(v4 + 16LL * v13);
  v15 = (void *)*result;
  if ( (_UNKNOWN *)*result == &unk_5040920 )
  {
LABEL_5:
    v16 = result + 1;
    if ( result[1] )
      return result;
    goto LABEL_46;
  }
  while ( v15 != (void *)-4096LL )
  {
    if ( !v12 && v15 == (void *)-8192LL )
      v12 = result;
    v13 = (v3 - 1) & (v11 + v13);
    result = (_QWORD *)(v4 + 16LL * v13);
    v15 = (void *)*result;
    if ( (_UNKNOWN *)*result == &unk_5040920 )
      goto LABEL_5;
    ++v11;
  }
  if ( !v12 )
    v12 = result;
  v29 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v25 = v29 + 1;
  if ( 4 * (v29 + 1) >= 3 * v3 )
  {
LABEL_26:
    sub_2275E10(a2, 2 * v3);
    v21 = *(_DWORD *)(a2 + 24);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a2 + 8);
      v24 = (v21 - 1) & (((unsigned int)&unk_5040920 >> 9) ^ ((unsigned int)&unk_5040920 >> 4));
      v25 = *(_DWORD *)(a2 + 16) + 1;
      v12 = (_QWORD *)(v23 + 16LL * v24);
      v26 = (void *)*v12;
      if ( (_UNKNOWN *)*v12 != &unk_5040920 )
      {
        v27 = 1;
        v28 = 0;
        while ( v26 != (void *)-4096LL )
        {
          if ( !v28 && v26 == (void *)-8192LL )
            v28 = v12;
          v24 = v22 & (v27 + v24);
          v12 = (_QWORD *)(v23 + 16LL * v24);
          v26 = (void *)*v12;
          if ( (_UNKNOWN *)*v12 == &unk_5040920 )
            goto LABEL_43;
          ++v27;
        }
        if ( v28 )
          v12 = v28;
      }
      goto LABEL_43;
    }
    goto LABEL_91;
  }
  if ( v3 - *(_DWORD *)(a2 + 20) - v25 <= v3 >> 3 )
  {
    sub_2275E10(a2, v3);
    v45 = *(_DWORD *)(a2 + 24);
    if ( v45 )
    {
      v46 = v45 - 1;
      v47 = *(_QWORD *)(a2 + 8);
      v48 = 1;
      v49 = v46 & (((unsigned int)&unk_5040920 >> 9) ^ ((unsigned int)&unk_5040920 >> 4));
      v50 = 0;
      v25 = *(_DWORD *)(a2 + 16) + 1;
      v12 = (_QWORD *)(v47 + 16LL * v49);
      v51 = (void *)*v12;
      if ( (_UNKNOWN *)*v12 != &unk_5040920 )
      {
        while ( v51 != (void *)-4096LL )
        {
          if ( !v50 && v51 == (void *)-8192LL )
            v50 = v12;
          v49 = v46 & (v48 + v49);
          v12 = (_QWORD *)(v47 + 16LL * v49);
          v51 = (void *)*v12;
          if ( (_UNKNOWN *)*v12 == &unk_5040920 )
            goto LABEL_43;
          ++v48;
        }
        if ( v50 )
          v12 = v50;
      }
      goto LABEL_43;
    }
LABEL_91:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
LABEL_43:
  *(_DWORD *)(a2 + 16) = v25;
  if ( *v12 != -4096 )
    --*(_DWORD *)(a2 + 20);
  *v12 = &unk_5040920;
  v16 = v12 + 1;
  v12[1] = 0;
LABEL_46:
  result = (_QWORD *)sub_22077B0(0x10u);
  if ( result )
    *result = &unk_4A31120;
  v30 = *v16;
  *v16 = (__int64)result;
  if ( v30 )
    return (_QWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
  return result;
}
