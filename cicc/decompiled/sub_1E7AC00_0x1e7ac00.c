// Function: sub_1E7AC00
// Address: 0x1e7ac00
//
_QWORD *__fastcall sub_1E7AC00(_QWORD *a1, __int64 a2)
{
  _QWORD *v4; // rax
  unsigned int v5; // r12d
  _QWORD *v6; // r13
  void *v7; // r15
  _QWORD *v8; // r12
  _QWORD *v9; // r8
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  _BOOL8 v13; // rdi
  __int64 v15; // rax
  unsigned __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rax
  const void *v19; // rsi
  int v20; // eax
  void *v21; // r9
  size_t v22; // rdx
  _QWORD *v23; // [rsp+8h] [rbp-38h]

  v4 = (_QWORD *)sub_22077B0(88);
  v5 = *(_DWORD *)(a2 + 16);
  v6 = v4;
  v7 = v4 + 7;
  v4[4] = *(_QWORD *)a2;
  v4[5] = v4 + 7;
  v4[6] = 0x400000000LL;
  if ( v5 )
  {
    v17 = (__int64)(v4 + 5);
    if ( v4 + 5 != (_QWORD *)(a2 + 8) )
    {
      v18 = *(_QWORD *)(a2 + 8);
      v19 = (const void *)(a2 + 24);
      if ( v18 != a2 + 24 )
      {
        v6[5] = v18;
        v20 = *(_DWORD *)(a2 + 20);
        v9 = a1 + 1;
        *((_DWORD *)v6 + 12) = v5;
        *(_QWORD *)(a2 + 8) = v19;
        *(_QWORD *)(a2 + 16) = 0;
        v8 = (_QWORD *)a1[2];
        *((_DWORD *)v6 + 13) = v20;
        if ( !v8 )
          goto LABEL_22;
        goto LABEL_3;
      }
      v21 = v7;
      v22 = 8LL * v5;
      if ( v5 <= 4
        || (sub_16CD150(v17, v7, v5, 8, v5, (int)v7),
            v21 = (void *)v6[5],
            v19 = *(const void **)(a2 + 8),
            (v22 = 8LL * *(unsigned int *)(a2 + 16)) != 0) )
      {
        memcpy(v21, v19, v22);
      }
      *((_DWORD *)v6 + 12) = v5;
      *(_DWORD *)(a2 + 16) = 0;
    }
  }
  v8 = (_QWORD *)a1[2];
  v9 = a1 + 1;
  if ( !v8 )
  {
LABEL_22:
    v8 = v9;
    if ( v9 == (_QWORD *)a1[3] )
    {
      v13 = 1;
LABEL_12:
      sub_220F040(v13, v6, v8, v9);
      ++a1[5];
      return v6;
    }
    v10 = v6[4];
LABEL_14:
    v23 = v9;
    v15 = sub_220EF80(v8);
    v9 = v23;
    if ( v10 <= *(_QWORD *)(v15 + 32) )
    {
      v8 = (_QWORD *)v15;
      goto LABEL_16;
    }
LABEL_10:
    v13 = 1;
    if ( v9 != v8 )
      v13 = v10 < v8[4];
    goto LABEL_12;
  }
LABEL_3:
  v10 = v6[4];
  while ( 1 )
  {
    v11 = v8[4];
    v12 = (_QWORD *)v8[3];
    if ( v10 < v11 )
      v12 = (_QWORD *)v8[2];
    if ( !v12 )
      break;
    v8 = v12;
  }
  if ( v10 < v11 )
  {
    if ( v8 == (_QWORD *)a1[3] )
      goto LABEL_10;
    goto LABEL_14;
  }
  if ( v10 > v11 )
    goto LABEL_10;
LABEL_16:
  v16 = v6[5];
  if ( v7 != (void *)v16 )
    _libc_free(v16);
  j_j___libc_free_0(v6, 88);
  return v8;
}
