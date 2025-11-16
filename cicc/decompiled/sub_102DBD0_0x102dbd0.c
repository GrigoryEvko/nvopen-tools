// Function: sub_102DBD0
// Address: 0x102dbd0
//
__int64 __fastcall sub_102DBD0(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  __int64 v6; // rdi
  __int64 v7; // r9
  _QWORD *v8; // r12
  unsigned int v9; // ecx
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // r12
  __int64 result; // rax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rbx
  _BYTE *v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rax
  signed __int64 v20; // r15
  char *v21; // rsi
  _BYTE *v22; // rdx
  int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  unsigned __int64 v27; // rax
  void *v28; // rcx
  void *v29; // rax
  _BYTE *v30; // rdi
  int v31; // eax
  int v32; // esi
  unsigned int v33; // eax
  __int64 v34; // rcx
  _QWORD *v35; // rdi
  int v36; // eax
  int v37; // edi
  int v38; // ecx
  __int64 v39; // r13
  __int64 v40; // rsi
  _QWORD *v41; // rax
  __int64 v42; // rax
  __int64 v43; // [rsp+0h] [rbp-150h]
  void *src; // [rsp+10h] [rbp-140h] BYREF
  __int64 v45; // [rsp+18h] [rbp-138h]
  _BYTE v46[304]; // [rsp+20h] [rbp-130h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_44;
  }
  v5 = v4 - 1;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v6 + 24LL * v9);
  v11 = *v10;
  if ( *v10 != a2 )
  {
    while ( v11 != -4096 )
    {
      if ( !v8 && v11 == -8192 )
        v8 = v10;
      v9 = v5 & (v7 + v9);
      v10 = (_QWORD *)(v6 + 24LL * v9);
      v11 = *v10;
      if ( *v10 == a2 )
        goto LABEL_3;
      v7 = (unsigned int)(v7 + 1);
    }
    if ( !v8 )
      v8 = v10;
    v14 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 20) - v15 > v4 >> 3 )
      {
LABEL_16:
        *(_DWORD *)(a1 + 16) = v15;
        if ( *v8 != -4096 )
          --*(_DWORD *)(a1 + 20);
        *v8 = a2;
        v12 = v8 + 1;
        *v12 = 0;
        v12[1] = 0;
        goto LABEL_19;
      }
      sub_102D9D0(a1, v4);
      v36 = *(_DWORD *)(a1 + 24);
      if ( v36 )
      {
        v37 = v36 - 1;
        v5 = *(_QWORD *)(a1 + 8);
        v38 = 1;
        LODWORD(v39) = (v36 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v8 = (_QWORD *)(v5 + 24LL * (unsigned int)v39);
        v40 = *v8;
        v15 = *(_DWORD *)(a1 + 16) + 1;
        v41 = 0;
        if ( *v8 != a2 )
        {
          while ( v40 != -4096 )
          {
            if ( !v41 && v40 == -8192 )
              v41 = v8;
            v7 = (unsigned int)(v38 + 1);
            v39 = v37 & (unsigned int)(v39 + v38);
            v8 = (_QWORD *)(v5 + 24 * v39);
            v40 = *v8;
            if ( *v8 == a2 )
              goto LABEL_16;
            ++v38;
          }
          if ( v41 )
            v8 = v41;
        }
        goto LABEL_16;
      }
LABEL_68:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
LABEL_44:
    sub_102D9D0(a1, 2 * v4);
    v31 = *(_DWORD *)(a1 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v7 = *(_QWORD *)(a1 + 8);
      v33 = (v31 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v8 = (_QWORD *)(v7 + 24LL * v33);
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v34 = *v8;
      if ( *v8 != a2 )
      {
        v5 = 1;
        v35 = 0;
        while ( v34 != -4096 )
        {
          if ( !v35 && v34 == -8192 )
            v35 = v8;
          v33 = v32 & (v5 + v33);
          v8 = (_QWORD *)(v7 + 24LL * v33);
          v34 = *v8;
          if ( *v8 == a2 )
            goto LABEL_16;
          v5 = (unsigned int)(v5 + 1);
        }
        if ( v35 )
          v8 = v35;
      }
      goto LABEL_16;
    }
    goto LABEL_68;
  }
LABEL_3:
  v12 = v10 + 1;
  if ( v10[1] )
    return v10[1];
LABEL_19:
  v16 = *(_QWORD *)(a2 + 16);
  if ( v16 )
  {
    while ( 1 )
    {
      v17 = *(_BYTE **)(v16 + 24);
      if ( (unsigned __int8)(*v17 - 30) <= 0xAu )
        break;
      v16 = *(_QWORD *)(v16 + 8);
      if ( !v16 )
        goto LABEL_42;
    }
    v18 = 0;
    src = v46;
    v45 = 0x2000000000LL;
    v19 = v16;
    while ( 1 )
    {
      v19 = *(_QWORD *)(v19 + 8);
      if ( !v19 )
        break;
      while ( (unsigned __int8)(**(_BYTE **)(v19 + 24) - 30) <= 0xAu )
      {
        v19 = *(_QWORD *)(v19 + 8);
        ++v18;
        if ( !v19 )
          goto LABEL_25;
      }
    }
LABEL_25:
    v20 = v18 + 1;
    v21 = v46;
    if ( v20 > 32 )
    {
      sub_C8D5F0((__int64)&src, v46, v20, 8u, v5, v7);
      v17 = *(_BYTE **)(v16 + 24);
      v21 = (char *)src + 8 * (unsigned int)v45;
    }
    v22 = v17;
LABEL_30:
    if ( v21 )
      *(_QWORD *)v21 = *((_QWORD *)v22 + 5);
    while ( 1 )
    {
      v16 = *(_QWORD *)(v16 + 8);
      if ( !v16 )
        break;
      v22 = *(_BYTE **)(v16 + 24);
      if ( (unsigned __int8)(*v22 - 30) <= 0xAu )
      {
        v21 += 8;
        goto LABEL_30;
      }
    }
    v23 = v20 + v45;
    v24 = (unsigned int)(v20 + v45);
    v25 = 8 * v24;
  }
  else
  {
LABEL_42:
    v25 = 0;
    v24 = 0;
    v23 = 0;
    src = v46;
    HIDWORD(v45) = 32;
  }
  v26 = *(_QWORD *)(a1 + 32);
  LODWORD(v45) = v23;
  *(_QWORD *)(a1 + 112) += v25;
  v27 = (v26 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(a1 + 40) >= v27 + v25 && v26 )
  {
    *(_QWORD *)(a1 + 32) = v27 + v25;
    v28 = (void *)((v26 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  }
  else
  {
    v42 = sub_9D1E70(a1 + 32, v25, v25, 3);
    v24 = (unsigned int)v45;
    v28 = (void *)v42;
    v25 = 8LL * (unsigned int)v45;
  }
  if ( v25 )
  {
    v29 = memmove(v28, src, v25);
    v24 = (unsigned int)v45;
    v28 = v29;
  }
  *v12 = (__int64)v28;
  v30 = src;
  v12[1] = v24;
  result = *v12;
  if ( v30 != v46 )
  {
    v43 = *v12;
    _libc_free(v30, v24);
    return v43;
  }
  return result;
}
