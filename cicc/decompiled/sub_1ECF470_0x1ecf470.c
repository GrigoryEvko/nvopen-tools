// Function: sub_1ECF470
// Address: 0x1ecf470
//
__int64 __fastcall sub_1ECF470(__int64 *a1, char *a2, __int64 *a3)
{
  char *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rcx
  bool v8; // cf
  unsigned __int64 v9; // rax
  char *v10; // rsi
  __int64 v11; // rcx
  char *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // rsi
  char *v22; // r15
  __int64 i; // r14
  __int64 v24; // rcx
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rcx
  __int64 v28; // rcx
  volatile signed __int32 *v29; // rdi
  __int64 v30; // rdi
  volatile signed __int32 *v31; // rdi
  __int64 v32; // rdi
  char *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rsi
  char *v36; // rdi
  __int64 v38; // rcx
  __int64 *v39; // [rsp+8h] [rbp-58h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  __int64 v41; // [rsp+18h] [rbp-48h]
  char *v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+28h] [rbp-38h]

  v4 = (char *)a1[1];
  v42 = (char *)*a1;
  v5 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)&v4[-*a1] >> 3);
  if ( v5 == 0x1745D1745D1745DLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0x2E8BA2E8BA2E8BA3LL * ((a1[1] - *a1) >> 3);
  v8 = __CFADD__(v6, v5);
  v9 = v6 + v5;
  v10 = (char *)(a2 - v42);
  if ( v8 )
  {
    v38 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v40 = 0;
      v11 = 88;
      v44 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x1745D1745D1745DLL )
      v9 = 0x1745D1745D1745DLL;
    v38 = 88 * v9;
  }
  v39 = a3;
  v41 = v38;
  v44 = sub_22077B0(v38);
  a3 = v39;
  v40 = v44 + v41;
  v11 = v44 + 88;
LABEL_7:
  v12 = &v10[v44];
  if ( &v10[v44] )
  {
    v13 = *a3;
    v14 = a3[2];
    *a3 = 0;
    *(_QWORD *)v12 = v13;
    v15 = a3[1];
    *((_QWORD *)v12 + 2) = v14;
    *((_QWORD *)v12 + 1) = v15;
    LODWORD(v15) = *((_DWORD *)a3 + 6);
    a3[1] = 0;
    *((_DWORD *)v12 + 6) = v15;
    v16 = a3[4];
    a3[4] = 0;
    *((_QWORD *)v12 + 4) = v16;
    *((_DWORD *)v12 + 10) = *((_DWORD *)a3 + 10);
    v17 = a3[6];
    a3[6] = 0;
    *((_QWORD *)v12 + 6) = v17;
    v18 = a3[7];
    a3[7] = 0;
    *((_QWORD *)v12 + 7) = v18;
    v19 = a3[8];
    a3[8] = 0;
    *((_QWORD *)v12 + 8) = v19;
    v20 = a3[9];
    a3[9] = 0;
    *((_QWORD *)v12 + 9) = v20;
    v21 = a3[10];
    a3[10] = 0;
    *((_QWORD *)v12 + 10) = v21;
  }
  v22 = v42;
  if ( a2 != v42 )
  {
    for ( i = v44; ; i += 88 )
    {
      if ( i )
      {
        v24 = *(_QWORD *)v22;
        *(_QWORD *)(i + 8) = 0;
        *(_QWORD *)i = v24;
        v25 = *((_QWORD *)v22 + 1);
        *((_QWORD *)v22 + 1) = 0;
        *(_QWORD *)(i + 8) = v25;
        LODWORD(v25) = *((_DWORD *)v22 + 4);
        *(_QWORD *)v22 = 0;
        *(_DWORD *)(i + 16) = v25;
        *(_DWORD *)(i + 20) = *((_DWORD *)v22 + 5);
        *(_DWORD *)(i + 24) = *((_DWORD *)v22 + 6);
        *(_QWORD *)(i + 32) = *((_QWORD *)v22 + 4);
        LODWORD(v25) = *((_DWORD *)v22 + 10);
        *((_QWORD *)v22 + 4) = 0;
        *(_DWORD *)(i + 40) = v25;
        v26 = *((_QWORD *)v22 + 6);
        *(_QWORD *)(i + 56) = 0;
        *(_QWORD *)(i + 48) = v26;
        v27 = *((_QWORD *)v22 + 7);
        *((_QWORD *)v22 + 7) = 0;
        *(_QWORD *)(i + 56) = v27;
        v28 = *((_QWORD *)v22 + 8);
        *((_QWORD *)v22 + 6) = 0;
        *(_QWORD *)(i + 64) = v28;
        *(_QWORD *)(i + 72) = *((_QWORD *)v22 + 9);
        *(_QWORD *)(i + 80) = *((_QWORD *)v22 + 10);
        *((_QWORD *)v22 + 10) = 0;
        *((_QWORD *)v22 + 8) = 0;
      }
      else
      {
        v32 = *((_QWORD *)v22 + 8);
        if ( v32 )
          j_j___libc_free_0(v32, *((_QWORD *)v22 + 10) - v32);
      }
      v29 = (volatile signed __int32 *)*((_QWORD *)v22 + 7);
      if ( v29 )
        sub_A191D0(v29);
      v30 = *((_QWORD *)v22 + 4);
      if ( v30 )
        j_j___libc_free_0_0(v30);
      v31 = (volatile signed __int32 *)*((_QWORD *)v22 + 1);
      if ( v31 )
        sub_A191D0(v31);
      v22 += 88;
      if ( v22 == a2 )
        break;
    }
    v11 = i + 176;
  }
  if ( a2 != v4 )
  {
    v33 = a2;
    v34 = v11;
    do
    {
      v35 = *(_QWORD *)v33;
      v33 += 88;
      v34 += 88;
      *(_QWORD *)(v34 - 88) = v35;
      *(_QWORD *)(v34 - 80) = *((_QWORD *)v33 - 10);
      *(_DWORD *)(v34 - 72) = *((_DWORD *)v33 - 18);
      *(_DWORD *)(v34 - 68) = *((_DWORD *)v33 - 17);
      *(_DWORD *)(v34 - 64) = *((_DWORD *)v33 - 16);
      *(_QWORD *)(v34 - 56) = *((_QWORD *)v33 - 7);
      *(_DWORD *)(v34 - 48) = *((_DWORD *)v33 - 12);
      *(_QWORD *)(v34 - 40) = *((_QWORD *)v33 - 5);
      *(_QWORD *)(v34 - 32) = *((_QWORD *)v33 - 4);
      *(_QWORD *)(v34 - 24) = *((_QWORD *)v33 - 3);
      *(_QWORD *)(v34 - 16) = *((_QWORD *)v33 - 2);
      *(_QWORD *)(v34 - 8) = *((_QWORD *)v33 - 1);
    }
    while ( v33 != v4 );
    v11 += 88 * (((0xE8BA2E8BA2E8BA3LL * ((unsigned __int64)(v33 - a2 - 88) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 1);
  }
  v36 = v42;
  if ( v42 )
  {
    v43 = v11;
    j_j___libc_free_0(v36, a1[2] - (_QWORD)v36);
    v11 = v43;
  }
  a1[1] = v11;
  *a1 = v44;
  a1[2] = v40;
  return v40;
}
