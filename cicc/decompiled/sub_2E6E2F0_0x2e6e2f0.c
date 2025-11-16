// Function: sub_2E6E2F0
// Address: 0x2e6e2f0
//
__int64 __fastcall sub_2E6E2F0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  unsigned __int64 v6; // r9
  __int64 v7; // r12
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // r8
  unsigned __int64 v11; // rcx
  __int64 v12; // rdx
  unsigned __int64 v13; // r13
  unsigned __int64 v14; // rdi
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // r13
  _QWORD *v19; // rax
  _QWORD *v20; // r13
  unsigned __int64 v21; // rax
  __int64 v22; // r13
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+18h] [rbp-38h]
  unsigned __int64 v28; // [rsp+18h] [rbp-38h]
  unsigned __int64 v29; // [rsp+18h] [rbp-38h]

  v5 = (_QWORD *)sub_22077B0(0x50u);
  v7 = (__int64)v5;
  if ( v5 )
  {
    *v5 = a2;
    v5[1] = a3;
    v8 = 0;
    if ( a3 )
      v8 = *(_DWORD *)(a3 + 16) + 1;
    *(_DWORD *)(v7 + 16) = v8;
    *(_QWORD *)(v7 + 24) = v7 + 40;
    *(_QWORD *)(v7 + 32) = 0x400000000LL;
    *(_QWORD *)(v7 + 72) = -1;
  }
  if ( a2 )
  {
    v9 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
    v10 = 8 * v9;
  }
  else
  {
    v10 = 0;
    LODWORD(v9) = 0;
  }
  v11 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)v11 > (unsigned int)v9 )
    goto LABEL_8;
  v16 = (unsigned int)(v9 + 1);
  v17 = (__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 104) + 104LL) - *(_QWORD *)(*(_QWORD *)(a1 + 104) + 96LL)) >> 3;
  if ( (unsigned int)v16 > (unsigned int)v17 )
    LODWORD(v17) = v16;
  if ( (unsigned int)v17 == v11 )
  {
LABEL_8:
    v12 = *(_QWORD *)(a1 + 24);
    goto LABEL_9;
  }
  v18 = 8LL * (unsigned int)v17;
  if ( (unsigned int)v17 < v11 )
  {
    v12 = *(_QWORD *)(a1 + 24);
    v21 = v12 + 8 * v11;
    v22 = v12 + v18;
    if ( v21 == v22 )
      goto LABEL_27;
    do
    {
      v6 = *(_QWORD *)(v21 - 8);
      v21 -= 8LL;
      if ( v6 )
      {
        v23 = *(_QWORD *)(v6 + 24);
        if ( v23 != v6 + 40 )
        {
          v24 = v6;
          v25 = v10;
          v28 = v21;
          _libc_free(v23);
          v6 = v24;
          v10 = v25;
          v21 = v28;
        }
        v26 = v10;
        v29 = v21;
        j_j___libc_free_0(v6);
        v10 = v26;
        v21 = v29;
      }
    }
    while ( v22 != v21 );
  }
  else
  {
    if ( (unsigned int)v17 > (unsigned __int64)*(unsigned int *)(a1 + 36) )
    {
      v27 = v10;
      sub_239B9C0(a1 + 24, (unsigned int)v17, v16, v11, v10, v6);
      v11 = *(unsigned int *)(a1 + 32);
      v10 = v27;
    }
    v12 = *(_QWORD *)(a1 + 24);
    v19 = (_QWORD *)(v12 + 8 * v11);
    v20 = (_QWORD *)(v12 + v18);
    if ( v19 == v20 )
      goto LABEL_27;
    do
    {
      if ( v19 )
        *v19 = 0;
      ++v19;
    }
    while ( v20 != v19 );
  }
  v12 = *(_QWORD *)(a1 + 24);
LABEL_27:
  *(_DWORD *)(a1 + 32) = v17;
LABEL_9:
  v13 = *(_QWORD *)(v12 + v10);
  *(_QWORD *)(v12 + v10) = v7;
  if ( v13 )
  {
    v14 = *(_QWORD *)(v13 + 24);
    if ( v14 != v13 + 40 )
      _libc_free(v14);
    j_j___libc_free_0(v13);
  }
  if ( a3 )
    sub_2E6D8E0(a3 + 24, v7, v12, v11, v10, v6);
  return v7;
}
