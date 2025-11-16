// Function: sub_2B52BE0
// Address: 0x2b52be0
//
_QWORD *__fastcall sub_2B52BE0(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // ebx
  const void *v8; // rsi
  _QWORD *v9; // rax
  __int64 i; // rdx
  unsigned __int64 v11; // r12
  int v12; // r13d
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  unsigned int v15; // esi
  unsigned int v16; // r13d
  unsigned int v17; // eax
  unsigned int v18; // r12d
  __int64 v19; // rbx
  __int64 v20; // rax
  unsigned int v21; // esi
  _BYTE *v22; // rax
  _BYTE *v23; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdx
  _BYTE *v28; // rdx
  unsigned __int64 v30; // r12
  unsigned __int64 v31; // rdx
  __int64 v34; // [rsp+18h] [rbp-98h]
  void *src; // [rsp+40h] [rbp-70h] BYREF
  __int64 v36; // [rsp+48h] [rbp-68h]
  _BYTE v37[96]; // [rsp+50h] [rbp-60h] BYREF

  v7 = a5;
  v8 = a1 + 2;
  v34 = (unsigned int)a5;
  *a1 = a1 + 2;
  a1[1] = 0x600000000LL;
  if ( (_DWORD)a5 )
  {
    v9 = a1 + 2;
    if ( (unsigned int)a5 > 6uLL )
    {
      sub_C8D5F0((__int64)a1, v8, (unsigned int)a5, 8u, a5, a6);
      v9 = (_QWORD *)(*a1 + 8LL * *((unsigned int *)a1 + 2));
      for ( i = *a1 + 8 * v34; (_QWORD *)i != v9; ++v9 )
      {
LABEL_4:
        if ( v9 )
          *v9 = 0;
      }
    }
    else
    {
      i = (__int64)v8 + 8 * (unsigned int)a5;
      if ( v8 != (const void *)i )
        goto LABEL_4;
    }
    *((_DWORD *)a1 + 2) = v7;
  }
  v11 = *((unsigned int *)a3 + 2);
  v12 = v11;
  if ( *(_DWORD *)(a4 + 12) < (unsigned int)v11 )
  {
    v31 = *((unsigned int *)a3 + 2);
    *(_DWORD *)(a4 + 8) = 0;
    sub_C8D5F0(a4, (const void *)(a4 + 16), v31, 4u, a5, a6);
    memset(*(void **)a4, 255, 4 * v11);
    *(_DWORD *)(a4 + 8) = v11;
  }
  else
  {
    v13 = *((unsigned int *)a3 + 2);
    v14 = *(unsigned int *)(a4 + 8);
    if ( v14 <= v11 )
      v13 = *(unsigned int *)(a4 + 8);
    if ( v13 )
    {
      memset(*(void **)a4, 255, 4 * v13);
      v14 = *(unsigned int *)(a4 + 8);
    }
    if ( v14 < v11 )
    {
      v30 = v11 - v14;
      if ( v30 )
      {
        if ( 4 * v30 )
          memset((void *)(*(_QWORD *)a4 + 4 * v14), 255, 4 * v30);
      }
    }
    *(_DWORD *)(a4 + 8) = v12;
  }
  v15 = *((_DWORD *)a3 + 2);
  v16 = 1;
  v17 = (v15 != 0) + (v15 - (v15 != 0)) / v7;
  if ( v17 > 1 )
  {
    _BitScanReverse(&v17, v17 - 1);
    v16 = 1 << (32 - (v17 ^ 0x1F));
  }
  if ( v15 <= v16 )
    v16 = *((_DWORD *)a3 + 2);
  if ( v34 )
  {
    v18 = 0;
    v19 = 0;
    while ( 1 )
    {
      v20 = *a3;
      v21 = v15 - v18;
      if ( v21 > v16 )
        v21 = v16;
      src = v37;
      v36 = 0xC00000000LL;
      *(_QWORD *)(*a1 + 8LL * (unsigned int)v19) = sub_2B52070(
                                                     a2,
                                                     (__int64 *)(v20 + 8LL * v18),
                                                     v21,
                                                     (__int64)&src,
                                                     v18,
                                                     v20 + 8LL * v18);
      v22 = src;
      if ( 4LL * (unsigned int)v36 )
      {
        memmove((void *)(*(_QWORD *)a4 + 4LL * v18), src, 4LL * (unsigned int)v36);
        v22 = src;
      }
      if ( v22 != v37 )
        _libc_free((unsigned __int64)v22);
      ++v19;
      v18 += v16;
      if ( v34 == v19 )
        break;
      v15 = *((_DWORD *)a3 + 2);
    }
  }
  v23 = (_BYTE *)*a1;
  v24 = 8LL * *((unsigned int *)a1 + 2);
  v25 = (_BYTE *)(*a1 + v24);
  v26 = v24 >> 3;
  v27 = v24 >> 5;
  if ( v27 )
  {
    v28 = &v23[32 * v27];
    while ( !v23[4] )
    {
      if ( v23[12] )
      {
        v23 += 8;
        goto LABEL_36;
      }
      if ( v23[20] )
      {
        v23 += 16;
        goto LABEL_36;
      }
      if ( v23[28] )
      {
        v23 += 24;
        goto LABEL_36;
      }
      v23 += 32;
      if ( v28 == v23 )
      {
        v26 = (v25 - v23) >> 3;
        goto LABEL_39;
      }
    }
    goto LABEL_36;
  }
LABEL_39:
  if ( v26 == 2 )
    goto LABEL_51;
  if ( v26 == 3 )
  {
    if ( v23[4] )
      goto LABEL_36;
    v23 += 8;
LABEL_51:
    if ( v23[4] )
      goto LABEL_36;
    v23 += 8;
    goto LABEL_53;
  }
  if ( v26 != 1 )
    goto LABEL_42;
LABEL_53:
  if ( !v23[4] )
    goto LABEL_42;
LABEL_36:
  if ( v25 == v23 )
LABEL_42:
    *((_DWORD *)a1 + 2) = 0;
  return a1;
}
