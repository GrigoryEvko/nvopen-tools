// Function: sub_2F96A60
// Address: 0x2f96a60
//
_QWORD *__fastcall sub_2F96A60(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // rdi
  _QWORD *v11; // rdx
  _QWORD *v12; // rcx
  _QWORD *v13; // rdi
  _QWORD *v14; // rsi
  _QWORD *v15; // r8
  _QWORD **v16; // r14
  _QWORD **v17; // rbx
  _QWORD *v18; // r15
  _QWORD **v19; // r12
  unsigned __int64 v20; // rdi
  int v21; // ebx
  _QWORD *v23; // [rsp+0h] [rbp-50h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  unsigned __int64 v25[7]; // [rsp+18h] [rbp-38h] BYREF

  v24 = a1 + 16;
  v7 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v25, a6);
  v8 = *(_QWORD *)a1;
  v23 = v7;
  v9 = 32LL * *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1 + v9;
  if ( v8 == v8 + v9 )
    goto LABEL_14;
  v11 = (_QWORD *)(v8 + 8);
  v12 = &v7[(unsigned __int64)v9 / 8];
  do
  {
    while ( 1 )
    {
      if ( !v7 )
        goto LABEL_4;
      *v7 = *(v11 - 1);
      v13 = (_QWORD *)*v11;
      v14 = v7 + 1;
      v7[1] = *v11;
      v15 = (_QWORD *)v11[1];
      v7[2] = v15;
      v7[3] = v11[2];
      if ( v11 == v13 )
        break;
      *v15 = v14;
      *(_QWORD *)(v7[1] + 8LL) = v14;
      v11[1] = v11;
      *v11 = v11;
      v11[2] = 0;
LABEL_4:
      v7 += 4;
      v11 += 4;
      if ( v12 == v7 )
        goto LABEL_8;
    }
    v7[2] = v14;
    v7 += 4;
    v11 += 4;
    *(v7 - 3) = v14;
  }
  while ( v12 != v7 );
LABEL_8:
  v16 = *(_QWORD ***)a1;
  v10 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v10 )
  {
    v17 = (_QWORD **)(v10 - 24);
    do
    {
      v18 = *v17;
      v19 = v17 - 1;
      while ( v18 != v17 )
      {
        v20 = (unsigned __int64)v18;
        v18 = (_QWORD *)*v18;
        j_j___libc_free_0(v20);
      }
      v17 -= 4;
    }
    while ( v19 != v16 );
    v10 = *(_QWORD *)a1;
  }
LABEL_14:
  v21 = v25[0];
  if ( v24 != v10 )
    _libc_free(v10);
  *(_DWORD *)(a1 + 12) = v21;
  *(_QWORD *)a1 = v23;
  return v23;
}
