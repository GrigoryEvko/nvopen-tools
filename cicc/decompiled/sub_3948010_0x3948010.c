// Function: sub_3948010
// Address: 0x3948010
//
unsigned __int64 __fastcall sub_3948010(unsigned __int64 *a1, char *a2, unsigned __int64 *a3)
{
  char *v6; // rbx
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  bool v10; // cf
  unsigned __int64 v11; // rax
  char *v12; // rdx
  unsigned __int64 v13; // r13
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  char *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rsi
  unsigned __int64 i; // r13
  __int64 v23; // rdi
  unsigned __int64 v25; // r13
  __int64 v26; // rax
  _QWORD **v27; // r8
  _QWORD *v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // r8
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // [rsp+0h] [rbp-60h]
  _QWORD **v37; // [rsp+8h] [rbp-58h]
  __int64 v38; // [rsp+10h] [rbp-50h]
  _QWORD **v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  unsigned __int64 v41; // [rsp+20h] [rbp-40h]
  unsigned __int64 v42; // [rsp+28h] [rbp-38h]

  v6 = (char *)a1[1];
  v7 = *a1;
  v8 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)&v6[-*a1] >> 3);
  if ( v8 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  if ( v8 )
    v9 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)&v6[-v7] >> 3);
  v10 = __CFADD__(v9, v8);
  v11 = v9 - 0x3333333333333333LL * ((__int64)&v6[-v7] >> 3);
  v12 = &a2[-v7];
  if ( v10 )
  {
    v25 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v11 )
    {
      v42 = 40;
      v36 = 0;
      v41 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x333333333333333LL )
      v11 = 0x333333333333333LL;
    v25 = 40 * v11;
  }
  v26 = sub_22077B0(v25);
  v12 = &a2[-v7];
  v41 = v26;
  v36 = v26 + v25;
  v42 = v26 + 40;
LABEL_7:
  v13 = *a3;
  *a3 = 0;
  v14 = (unsigned __int64 *)&v12[v41];
  if ( &v12[v41] )
  {
    *v14 = v13;
    v14[1] = 0;
    v14[2] = 0;
    v14[3] = 0x2800000000LL;
  }
  else if ( v13 )
  {
    v27 = *(_QWORD ***)(v13 + 120);
    v39 = *(_QWORD ***)(v13 + 128);
    if ( v39 != v27 )
    {
      do
      {
        v28 = *v27;
        if ( *v27 )
        {
          v37 = v27;
          sub_16C93F0(v28);
          j_j___libc_free_0((unsigned __int64)v28);
          v27 = v37;
        }
        v27 += 2;
      }
      while ( v39 != v27 );
      v27 = *(_QWORD ***)(v13 + 120);
    }
    if ( v27 )
      j_j___libc_free_0((unsigned __int64)v27);
    sub_3947930(v13 + 64);
    v29 = *(_QWORD *)(v13 + 64);
    if ( v29 != v13 + 112 )
      j_j___libc_free_0(v29);
    v30 = *(_QWORD *)(v13 + 40);
    if ( v30 )
      j_j___libc_free_0(v30);
    v31 = *(_QWORD *)v13;
    if ( *(_DWORD *)(v13 + 12) )
    {
      v32 = *(unsigned int *)(v13 + 8);
      if ( (_DWORD)v32 )
      {
        v33 = 8 * v32;
        v34 = 0;
        do
        {
          v35 = *(_QWORD *)(v31 + v34);
          if ( v35 != -8 && v35 )
          {
            v38 = v33;
            v40 = v34;
            _libc_free(v35);
            v31 = *(_QWORD *)v13;
            v33 = v38;
            v34 = v40;
          }
          v34 += 8;
        }
        while ( v33 != v34 );
      }
    }
    _libc_free(v31);
    j_j___libc_free_0(v13);
  }
  if ( a2 != (char *)v7 )
  {
    v15 = v41;
    v16 = v7;
    do
    {
      if ( v15 )
      {
        *(_QWORD *)v15 = *(_QWORD *)v16;
        v17 = *(_QWORD *)(v16 + 8);
        *(_QWORD *)v16 = 0;
        *(_QWORD *)(v15 + 8) = v17;
        *(_DWORD *)(v15 + 16) = *(_DWORD *)(v16 + 16);
        *(_DWORD *)(v15 + 20) = *(_DWORD *)(v16 + 20);
        *(_DWORD *)(v15 + 24) = *(_DWORD *)(v16 + 24);
        *(_DWORD *)(v15 + 28) = *(_DWORD *)(v16 + 28);
        *(_QWORD *)(v16 + 8) = 0;
        *(_DWORD *)(v16 + 16) = 0;
        *(_DWORD *)(v16 + 20) = 0;
        *(_DWORD *)(v16 + 24) = 0;
      }
      v16 += 40LL;
      v15 += 40LL;
    }
    while ( (char *)v16 != a2 );
    v42 = v41 + 8 * ((unsigned __int64)&a2[-v7 - 40] >> 3) + 80;
  }
  v18 = v42;
  v19 = a2;
  if ( a2 != v6 )
  {
    do
    {
      v20 = *(_QWORD *)v19;
      *(_QWORD *)v19 = 0;
      v19 += 40;
      v18 += 40LL;
      *(_QWORD *)(v18 - 40) = v20;
      v21 = *((_QWORD *)v19 - 4);
      *((_QWORD *)v19 - 4) = 0;
      *(_QWORD *)(v18 - 32) = v21;
      LODWORD(v21) = *((_DWORD *)v19 - 6);
      *((_DWORD *)v19 - 6) = 0;
      *(_DWORD *)(v18 - 24) = v21;
      LODWORD(v21) = *((_DWORD *)v19 - 5);
      *((_DWORD *)v19 - 5) = 0;
      *(_DWORD *)(v18 - 20) = v21;
      LODWORD(v21) = *((_DWORD *)v19 - 4);
      *((_DWORD *)v19 - 4) = 0;
      *(_DWORD *)(v18 - 16) = v21;
      *(_DWORD *)(v18 - 12) = *((_DWORD *)v19 - 3);
    }
    while ( v6 != v19 );
    v42 += 8 * ((unsigned __int64)(v6 - a2 - 40) >> 3) + 40;
  }
  for ( i = v7; (char *)i != v6; i += 40LL )
  {
    v23 = i;
    sub_3947D10(v23);
  }
  if ( v7 )
    j_j___libc_free_0(v7);
  *a1 = v41;
  a1[1] = v42;
  a1[2] = v36;
  return v36;
}
