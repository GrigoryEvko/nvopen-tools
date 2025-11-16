// Function: sub_29A62B0
// Address: 0x29a62b0
//
_QWORD *__fastcall sub_29A62B0(_QWORD *a1, __int64 *a2, __int64 *a3)
{
  __int64 v5; // rax
  unsigned __int64 *v6; // rdx
  unsigned __int64 v7; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned int v12; // r14d
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  int v15; // esi
  _QWORD *v16; // r8
  __int64 v17; // rax
  _QWORD *v18; // r12
  unsigned __int64 v19; // r14
  unsigned __int64 v20; // rdx
  _QWORD *v21; // rax
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // r15
  _QWORD *v24; // rdi
  unsigned __int64 v25; // rdi
  _QWORD *v26; // rax
  __int64 v27; // rax
  __int64 v29; // rax
  char v30; // di
  __int64 *v31; // rax
  const void *v32; // rsi
  void *v33; // r9
  size_t v34; // rdx
  int v35; // eax
  _QWORD *v36; // [rsp+8h] [rbp-38h]

  v5 = sub_22077B0(0x100u);
  v6 = (unsigned __int64 *)a3[1];
  v7 = v5;
  v8 = *a2;
  *(_QWORD *)(v7 + 48) = v6;
  v9 = v7 + 40;
  *(_QWORD *)(v7 + 32) = v8;
  v10 = *a3;
  *(_QWORD *)(v7 + 40) = *a3;
  if ( v6 )
  {
    *v6 = v9;
    v10 = *a3;
  }
  if ( v10 )
    *(_QWORD *)(v10 + 8) = v9;
  v11 = a3[2];
  v12 = *((_DWORD *)a3 + 8);
  a3[1] = 0;
  *a3 = 0;
  *(_QWORD *)(v7 + 56) = v11;
  *(_QWORD *)(v7 + 64) = v7 + 80;
  *(_QWORD *)(v7 + 72) = 0x1000000000LL;
  if ( v12 && (__int64 *)(v7 + 64) != a3 + 3 )
  {
    v31 = (__int64 *)a3[3];
    v32 = a3 + 5;
    if ( v31 == a3 + 5 )
    {
      v33 = (void *)(v7 + 80);
      v34 = 8LL * v12;
      if ( v12 <= 0x10
        || (sub_C8D5F0(v7 + 64, (const void *)(v7 + 80), v12, 8u, v12, (__int64)v33),
            v33 = *(void **)(v7 + 64),
            v32 = (const void *)a3[3],
            (v34 = 8LL * *((unsigned int *)a3 + 8)) != 0) )
      {
        memcpy(v33, v32, v34);
      }
      *(_DWORD *)(v7 + 72) = v12;
      *((_DWORD *)a3 + 8) = 0;
    }
    else
    {
      *(_QWORD *)(v7 + 64) = v31;
      v35 = *((_DWORD *)a3 + 9);
      *(_DWORD *)(v7 + 72) = v12;
      *(_DWORD *)(v7 + 76) = v35;
      a3[3] = (__int64)v32;
      a3[4] = 0;
    }
  }
  v13 = a3[23];
  v14 = v7 + 216;
  if ( !v13 )
  {
    v18 = (_QWORD *)a1[2];
    *(_QWORD *)(v7 + 232) = v14;
    v16 = a1 + 1;
    *(_DWORD *)(v7 + 216) = 0;
    *(_QWORD *)(v7 + 224) = 0;
    *(_QWORD *)(v7 + 240) = v14;
    *(_QWORD *)(v7 + 248) = 0;
    if ( v18 )
      goto LABEL_8;
LABEL_35:
    v18 = v16;
    if ( v16 == (_QWORD *)a1[3] )
    {
      v30 = 1;
LABEL_28:
      sub_220F040(v30, v7, v18, v16);
      ++a1[5];
      return (_QWORD *)v7;
    }
    v19 = *(_QWORD *)(v7 + 32);
    goto LABEL_25;
  }
  v15 = *((_DWORD *)a3 + 44);
  *(_QWORD *)(v7 + 224) = v13;
  v16 = a1 + 1;
  *(_DWORD *)(v7 + 216) = v15;
  *(_QWORD *)(v7 + 232) = a3[24];
  *(_QWORD *)(v7 + 240) = a3[25];
  *(_QWORD *)(v13 + 8) = v14;
  v17 = a3[26];
  a3[23] = 0;
  a3[24] = (__int64)(a3 + 22);
  a3[25] = (__int64)(a3 + 22);
  a3[26] = 0;
  v18 = (_QWORD *)a1[2];
  *(_QWORD *)(v7 + 248) = v17;
  if ( !v18 )
    goto LABEL_35;
LABEL_8:
  v19 = *(_QWORD *)(v7 + 32);
  while ( 1 )
  {
    v20 = v18[4];
    v21 = (_QWORD *)v18[3];
    if ( v19 < v20 )
      v21 = (_QWORD *)v18[2];
    if ( !v21 )
      break;
    v18 = v21;
  }
  if ( v19 < v20 )
  {
    if ( (_QWORD *)a1[3] == v18 )
    {
LABEL_26:
      v30 = 1;
      if ( v16 != v18 )
        v30 = v19 < v18[4];
      goto LABEL_28;
    }
LABEL_25:
    v36 = v16;
    v29 = sub_220EF80((__int64)v18);
    v16 = v36;
    if ( v19 <= *(_QWORD *)(v29 + 32) )
    {
      v18 = (_QWORD *)v29;
      goto LABEL_15;
    }
    goto LABEL_26;
  }
  if ( v19 > v20 )
    goto LABEL_26;
LABEL_15:
  v22 = *(_QWORD *)(v7 + 224);
  while ( v22 )
  {
    v23 = v22;
    sub_29A3730(*(_QWORD **)(v22 + 24));
    v24 = *(_QWORD **)(v22 + 56);
    v22 = *(_QWORD *)(v22 + 16);
    sub_29A3980(v24);
    j_j___libc_free_0(v23);
  }
  v25 = *(_QWORD *)(v7 + 64);
  if ( v7 + 80 != v25 )
    _libc_free(v25);
  v26 = *(_QWORD **)(v7 + 48);
  if ( v26 )
    *v26 = *(_QWORD *)(v7 + 40);
  v27 = *(_QWORD *)(v7 + 40);
  if ( v27 )
    *(_QWORD *)(v27 + 8) = *(_QWORD *)(v7 + 48);
  j_j___libc_free_0(v7);
  return v18;
}
