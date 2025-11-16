// Function: sub_1213000
// Address: 0x1213000
//
__int64 __fastcall sub_1213000(_QWORD *a1, _DWORD *a2, __int64 *a3)
{
  __int64 v5; // rax
  unsigned int v6; // r15d
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r12
  _QWORD *v11; // r15
  unsigned int v12; // ecx
  unsigned int v13; // edx
  __int64 v14; // rax
  _BOOL8 v15; // rdi
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 *v19; // rax
  void *v20; // r9
  size_t v21; // rdx
  int v22; // eax

  v5 = sub_22077B0(128);
  v6 = *((_DWORD *)a3 + 4);
  v7 = v5;
  v8 = v5 + 64;
  *(_DWORD *)(v5 + 32) = *a2;
  v9 = *a3;
  *(_QWORD *)(v7 + 48) = v7 + 64;
  *(_QWORD *)(v7 + 40) = v9;
  *(_QWORD *)(v7 + 56) = 0x800000000LL;
  if ( !v6 || (__int64 *)(v7 + 48) == a3 + 1 )
  {
LABEL_2:
    v10 = a1[2];
    v11 = a1 + 1;
    if ( !v10 )
      goto LABEL_24;
    goto LABEL_3;
  }
  v19 = (__int64 *)a3[1];
  a2 = a3 + 3;
  if ( v19 != a3 + 3 )
  {
    *(_QWORD *)(v7 + 48) = v19;
    v22 = *((_DWORD *)a3 + 5);
    *(_DWORD *)(v7 + 56) = v6;
    *(_DWORD *)(v7 + 60) = v22;
    a3[1] = (__int64)a2;
    a3[2] = 0;
    goto LABEL_2;
  }
  v20 = (void *)(v7 + 64);
  v21 = 8LL * v6;
  if ( v6 <= 8
    || (sub_C8D5F0(v7 + 48, (const void *)(v7 + 64), v6, 8u, v6, (__int64)v20),
        v20 = *(void **)(v7 + 48),
        a2 = (_DWORD *)a3[1],
        (v21 = 8LL * *((unsigned int *)a3 + 4)) != 0) )
  {
    memcpy(v20, a2, v21);
  }
  *((_DWORD *)a3 + 4) = 0;
  v10 = a1[2];
  *(_DWORD *)(v7 + 56) = v6;
  v11 = a1 + 1;
  if ( !v10 )
  {
LABEL_24:
    v10 = (__int64)v11;
    if ( v11 == (_QWORD *)a1[3] )
    {
      v15 = 1;
LABEL_12:
      sub_220F040(v15, v7, v10, v11);
      ++a1[5];
      return v7;
    }
LABEL_14:
    v17 = sub_220EF80(v10);
    if ( *(_DWORD *)(v17 + 32) >= *(_DWORD *)(v7 + 32) )
    {
      v10 = v17;
      goto LABEL_16;
    }
LABEL_10:
    v15 = 1;
    if ( v11 != (_QWORD *)v10 )
      v15 = *(_DWORD *)(v7 + 32) < *(_DWORD *)(v10 + 32);
    goto LABEL_12;
  }
LABEL_3:
  v12 = *(_DWORD *)(v7 + 32);
  while ( 1 )
  {
    v13 = *(_DWORD *)(v10 + 32);
    v14 = *(_QWORD *)(v10 + 24);
    if ( v12 < v13 )
      v14 = *(_QWORD *)(v10 + 16);
    LOBYTE(a2) = v12 < v13;
    if ( !v14 )
      break;
    v10 = v14;
  }
  if ( v12 < v13 )
  {
    if ( a1[3] == v10 )
      goto LABEL_10;
    goto LABEL_14;
  }
  if ( v12 > v13 )
    goto LABEL_10;
LABEL_16:
  v18 = *(_QWORD *)(v7 + 48);
  if ( v8 != v18 )
    _libc_free(v18, a2);
  j_j___libc_free_0(v7, 128);
  return v10;
}
