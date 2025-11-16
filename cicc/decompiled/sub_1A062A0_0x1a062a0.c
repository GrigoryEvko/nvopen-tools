// Function: sub_1A062A0
// Address: 0x1a062a0
//
unsigned __int64 __fastcall sub_1A062A0(__int64 a1, _QWORD *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r8
  _QWORD *v6; // r10
  int v7; // r11d
  unsigned __int64 result; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rcx
  int v11; // eax
  int v12; // edx
  _QWORD *v13; // rax
  char *v14; // r14
  char *v15; // rsi
  __int64 v16; // r13
  __int64 v17; // rcx
  __int64 v18; // r9
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rdi
  _QWORD *v21; // rax
  unsigned __int64 *v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // r15
  char *v25; // r14
  __int64 *v26; // r15
  size_t v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  int v30; // eax
  int v31; // ecx
  __int64 v32; // r8
  unsigned int v33; // eax
  __int64 v34; // rdi
  int v35; // r11d
  _QWORD *v36; // r9
  int v37; // eax
  int v38; // ecx
  __int64 v39; // r8
  int v40; // r11d
  unsigned int v41; // eax
  __int64 v42; // rdi
  __int64 v43; // rax
  unsigned __int64 v44; // r14
  __int64 v45; // rax
  const void *v46; // rsi
  __int64 v47; // rdx
  __int64 v48; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_30;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 0;
  v7 = 1;
  result = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v9 = (_QWORD *)(v5 + 8 * result);
  v10 = *v9;
  if ( *a2 == *v9 )
    return result;
  while ( v10 != -8 )
  {
    if ( v10 != -16 || v6 )
      v9 = v6;
    result = (v4 - 1) & (v7 + (_DWORD)result);
    v10 = *(_QWORD *)(v5 + 8LL * (unsigned int)result);
    if ( *a2 == v10 )
      return result;
    ++v7;
    v6 = v9;
    v9 = (_QWORD *)(v5 + 8LL * (unsigned int)result);
  }
  v11 = *(_DWORD *)(a1 + 16);
  if ( !v6 )
    v6 = v9;
  ++*(_QWORD *)a1;
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v4 )
  {
LABEL_30:
    sub_1A060F0(a1, 2 * v4);
    v30 = *(_DWORD *)(a1 + 24);
    if ( v30 )
    {
      v31 = v30 - 1;
      v32 = *(_QWORD *)(a1 + 8);
      v33 = (v30 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v6 = (_QWORD *)(v32 + 8LL * v33);
      v34 = *v6;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v6 == *a2 )
        goto LABEL_13;
      v35 = 1;
      v36 = 0;
      while ( v34 != -8 )
      {
        if ( v34 == -16 && !v36 )
          v36 = v6;
        v33 = v31 & (v35 + v33);
        v6 = (_QWORD *)(v32 + 8LL * v33);
        v34 = *v6;
        if ( *a2 == *v6 )
          goto LABEL_13;
        ++v35;
      }
LABEL_34:
      if ( v36 )
        v6 = v36;
      goto LABEL_13;
    }
LABEL_61:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v12 <= v4 >> 3 )
  {
    sub_1A060F0(a1, v4);
    v37 = *(_DWORD *)(a1 + 24);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a1 + 8);
      v36 = 0;
      v40 = 1;
      v41 = (v37 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v6 = (_QWORD *)(v39 + 8LL * v41);
      v42 = *v6;
      v12 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v6 == *a2 )
        goto LABEL_13;
      while ( v42 != -8 )
      {
        if ( !v36 && v42 == -16 )
          v36 = v6;
        v41 = v38 & (v40 + v41);
        v6 = (_QWORD *)(v39 + 8LL * v41);
        v42 = *v6;
        if ( *a2 == *v6 )
          goto LABEL_13;
        ++v40;
      }
      goto LABEL_34;
    }
    goto LABEL_61;
  }
LABEL_13:
  *(_DWORD *)(a1 + 16) = v12;
  if ( *v6 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v6 = *a2;
  v13 = *(_QWORD **)(a1 + 80);
  if ( v13 == (_QWORD *)(*(_QWORD *)(a1 + 96) - 8LL) )
  {
    v14 = *(char **)(a1 + 104);
    v15 = *(char **)(a1 + 72);
    v16 = v14 - v15;
    v17 = (v14 - v15) >> 3;
    if ( (((__int64)v13 - *(_QWORD *)(a1 + 88)) >> 3)
       + ((v17 - 1) << 6)
       + ((__int64)(*(_QWORD *)(a1 + 64) - *(_QWORD *)(a1 + 48)) >> 3) == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    v18 = *(_QWORD *)(a1 + 32);
    v19 = *(_QWORD *)(a1 + 40);
    v20 = v19 - ((__int64)&v14[-v18] >> 3);
    if ( v20 <= 1 )
    {
      v24 = v17 + 2;
      if ( v19 <= 2 * (v17 + 2) )
      {
        v43 = 1;
        if ( v19 )
          v43 = *(_QWORD *)(a1 + 40);
        v44 = v19 + v43 + 2;
        if ( v44 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(v20, v15, v19);
        v45 = sub_22077B0(8 * v44);
        v46 = *(const void **)(a1 + 72);
        v48 = v45;
        v26 = (__int64 *)(v45 + 8 * ((v44 - v24) >> 1));
        v47 = *(_QWORD *)(a1 + 104) + 8LL;
        if ( (const void *)v47 != v46 )
          memmove(v26, v46, v47 - (_QWORD)v46);
        j_j___libc_free_0(*(_QWORD *)(a1 + 32), 8LL * *(_QWORD *)(a1 + 40));
        *(_QWORD *)(a1 + 40) = v44;
        *(_QWORD *)(a1 + 32) = v48;
      }
      else
      {
        v25 = v14 + 8;
        v26 = (__int64 *)(v18 + 8 * ((v19 - v24) >> 1));
        v27 = v25 - v15;
        if ( v15 <= (char *)v26 )
        {
          if ( v15 != v25 )
            memmove((char *)v26 + v16 - v27 + 8, v15, v27);
        }
        else if ( v15 != v25 )
        {
          memmove(v26, v15, v27);
        }
      }
      *(_QWORD *)(a1 + 72) = v26;
      v28 = *v26;
      v14 = (char *)v26 + v16;
      *(_QWORD *)(a1 + 104) = (char *)v26 + v16;
      *(_QWORD *)(a1 + 56) = v28;
      *(_QWORD *)(a1 + 64) = v28 + 512;
      v29 = *(__int64 *)((char *)v26 + v16);
      *(_QWORD *)(a1 + 88) = v29;
      *(_QWORD *)(a1 + 96) = v29 + 512;
    }
    *((_QWORD *)v14 + 1) = sub_22077B0(512);
    v21 = *(_QWORD **)(a1 + 80);
    if ( v21 )
      *v21 = *a2;
    v22 = (unsigned __int64 *)(*(_QWORD *)(a1 + 104) + 8LL);
    *(_QWORD *)(a1 + 104) = v22;
    result = *v22;
    v23 = *v22 + 512;
    *(_QWORD *)(a1 + 88) = result;
    *(_QWORD *)(a1 + 96) = v23;
    *(_QWORD *)(a1 + 80) = result;
  }
  else
  {
    if ( v13 )
    {
      *v13 = *a2;
      v13 = *(_QWORD **)(a1 + 80);
    }
    result = (unsigned __int64)(v13 + 1);
    *(_QWORD *)(a1 + 80) = result;
  }
  return result;
}
