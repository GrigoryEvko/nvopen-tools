// Function: sub_1E3BA70
// Address: 0x1e3ba70
//
__int64 __fastcall sub_1E3BA70(__int64 *a1, _DWORD *a2, int *a3)
{
  int *v3; // rcx
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdx
  bool v7; // cf
  unsigned __int64 v8; // rax
  char *v9; // rbx
  char *v10; // rdi
  int v11; // eax
  char *v12; // rdx
  unsigned __int64 v13; // r14
  char *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // r15
  __int64 v19; // rax
  char *v20; // rdi
  size_t v21; // r14
  char *v22; // rax
  int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 i; // r15
  __int64 v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // r12
  __int64 v30; // r13
  volatile signed __int32 *v31; // rbx
  signed __int32 v32; // edx
  signed __int32 v33; // edx
  _DWORD *v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rdx
  int v37; // ecx
  int v38; // esi
  __int64 v39; // rcx
  int v40; // esi
  __int64 v41; // rcx
  int v42; // esi
  __int64 v43; // rcx
  __int64 result; // rax
  __int64 v45; // rdi
  __int64 v46; // rax
  int *v47; // [rsp+8h] [rbp-68h]
  int *v48; // [rsp+8h] [rbp-68h]
  int *v49; // [rsp+8h] [rbp-68h]
  int *v50; // [rsp+8h] [rbp-68h]
  __int64 v51; // [rsp+10h] [rbp-60h]
  __int64 v53; // [rsp+20h] [rbp-50h]
  __int64 v54; // [rsp+28h] [rbp-48h]
  _DWORD *v55; // [rsp+30h] [rbp-40h]

  v3 = a3;
  v4 = 0x1745D1745D1745DLL;
  v55 = (_DWORD *)a1[1];
  v53 = *a1;
  v5 = 0x2E8BA2E8BA2E8BA3LL * (((__int64)v55 - *a1) >> 3);
  if ( v5 == 0x1745D1745D1745DLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0x2E8BA2E8BA2E8BA3LL * (((__int64)v55 - *a1) >> 3);
  v7 = __CFADD__(v6, v5);
  v8 = v6 + v5;
  v51 = v8;
  if ( v7 )
  {
    v45 = 0x7FFFFFFFFFFFFFF8LL;
    v51 = 0x1745D1745D1745DLL;
  }
  else
  {
    if ( !v8 )
    {
      v54 = 0;
      goto LABEL_7;
    }
    if ( v8 <= 0x1745D1745D1745DLL )
      v4 = v8;
    v51 = v4;
    v45 = 88 * v4;
  }
  v50 = v3;
  v46 = sub_22077B0(v45);
  v3 = v50;
  v54 = v46;
LABEL_7:
  v9 = (char *)a2 + v54 - v53;
  if ( !v9 )
    goto LABEL_24;
  v10 = (char *)*((_QWORD *)v3 + 2);
  v11 = *v3;
  *((_QWORD *)v9 + 1) = 0;
  v12 = (char *)*((_QWORD *)v3 + 1);
  *((_QWORD *)v9 + 2) = 0;
  *(_DWORD *)v9 = v11;
  *((_QWORD *)v9 + 3) = 0;
  v13 = v10 - v12;
  if ( v10 == v12 )
  {
    v14 = 0;
  }
  else
  {
    if ( v13 > 0x7FFFFFFFFFFFFFF0LL )
      goto LABEL_63;
    v47 = v3;
    v14 = (char *)sub_22077B0(v10 - v12);
    v3 = v47;
    v10 = (char *)*((_QWORD *)v47 + 2);
    v12 = (char *)*((_QWORD *)v47 + 1);
  }
  *((_QWORD *)v9 + 1) = v14;
  *((_QWORD *)v9 + 2) = v14;
  *((_QWORD *)v9 + 3) = &v14[v13];
  if ( v10 == v12 )
  {
    v10 = v14;
  }
  else
  {
    v10 = &v14[v10 - v12];
    do
    {
      if ( v14 )
      {
        *(_QWORD *)v14 = *(_QWORD *)v12;
        v15 = *((_QWORD *)v12 + 1);
        *((_QWORD *)v14 + 1) = v15;
        if ( v15 )
        {
          if ( &_pthread_key_create )
            _InterlockedAdd((volatile signed __int32 *)(v15 + 8), 1u);
          else
            ++*(_DWORD *)(v15 + 8);
        }
      }
      v14 += 16;
      v12 += 16;
    }
    while ( v14 != v10 );
  }
  v16 = *((_QWORD *)v3 + 4);
  v4 = *((_QWORD *)v3 + 6);
  *((_QWORD *)v9 + 2) = v10;
  *((_QWORD *)v9 + 6) = 0;
  *((_QWORD *)v9 + 4) = v16;
  LODWORD(v16) = v3[10];
  *((_QWORD *)v9 + 7) = 0;
  *((_DWORD *)v9 + 10) = v16;
  v17 = *((_QWORD *)v3 + 7);
  *((_QWORD *)v9 + 8) = 0;
  v18 = v17 - v4;
  if ( v17 == v4 )
  {
    v21 = 0;
    v20 = 0;
    goto LABEL_21;
  }
  v48 = v3;
  if ( v18 > 0x7FFFFFFFFFFFFFFCLL )
LABEL_63:
    sub_4261EA(v10, v4, v12);
  v19 = sub_22077B0(v18);
  v3 = v48;
  v20 = (char *)v19;
  v17 = *((_QWORD *)v48 + 7);
  v4 = *((_QWORD *)v48 + 6);
  v21 = v17 - v4;
LABEL_21:
  *((_QWORD *)v9 + 6) = v20;
  *((_QWORD *)v9 + 7) = v20;
  *((_QWORD *)v9 + 8) = &v20[v18];
  if ( v4 != v17 )
  {
    v49 = v3;
    v22 = (char *)memmove(v20, (const void *)v4, v21);
    v3 = v49;
    v20 = v22;
  }
  v23 = v3[20];
  v24 = *((_QWORD *)v3 + 9);
  *((_QWORD *)v9 + 7) = &v20[v21];
  *((_QWORD *)v9 + 9) = v24;
  *((_DWORD *)v9 + 20) = v23;
LABEL_24:
  v25 = v53;
  for ( i = v54; (_DWORD *)v25 != a2; i += 88 )
  {
    if ( i )
    {
      *(_DWORD *)i = *(_DWORD *)v25;
      *(_QWORD *)(i + 8) = *(_QWORD *)(v25 + 8);
      *(_QWORD *)(i + 16) = *(_QWORD *)(v25 + 16);
      *(_QWORD *)(i + 24) = *(_QWORD *)(v25 + 24);
      v27 = *(_QWORD *)(v25 + 32);
      *(_QWORD *)(v25 + 24) = 0;
      *(_QWORD *)(v25 + 16) = 0;
      *(_QWORD *)(v25 + 8) = 0;
      *(_QWORD *)(i + 32) = v27;
      *(_DWORD *)(i + 40) = *(_DWORD *)(v25 + 40);
      *(_QWORD *)(i + 48) = *(_QWORD *)(v25 + 48);
      *(_QWORD *)(i + 56) = *(_QWORD *)(v25 + 56);
      *(_QWORD *)(i + 64) = *(_QWORD *)(v25 + 64);
      LODWORD(v27) = *(_DWORD *)(v25 + 72);
      *(_QWORD *)(v25 + 64) = 0;
      *(_QWORD *)(v25 + 56) = 0;
      *(_QWORD *)(v25 + 48) = 0;
      *(_DWORD *)(i + 72) = v27;
      *(_DWORD *)(i + 76) = *(_DWORD *)(v25 + 76);
      *(_DWORD *)(i + 80) = *(_DWORD *)(v25 + 80);
    }
    v28 = *(_QWORD *)(v25 + 48);
    if ( v28 )
    {
      v4 = *(_QWORD *)(v25 + 64) - v28;
      j_j___libc_free_0(v28, v4);
    }
    v29 = *(_QWORD *)(v25 + 16);
    v30 = *(_QWORD *)(v25 + 8);
    if ( v29 != v30 )
    {
      do
      {
        while ( 1 )
        {
          v31 = *(volatile signed __int32 **)(v30 + 8);
          if ( v31 )
          {
            if ( &_pthread_key_create )
            {
              v32 = _InterlockedExchangeAdd(v31 + 2, 0xFFFFFFFF);
            }
            else
            {
              v32 = *((_DWORD *)v31 + 2);
              *((_DWORD *)v31 + 2) = v32 - 1;
            }
            if ( v32 == 1 )
            {
              (*(void (__fastcall **)(volatile signed __int32 *, __int64, _QWORD, int *))(*(_QWORD *)v31 + 16LL))(
                v31,
                v4,
                *(_QWORD *)v31,
                v3);
              if ( &_pthread_key_create )
              {
                v33 = _InterlockedExchangeAdd(v31 + 3, 0xFFFFFFFF);
              }
              else
              {
                v33 = *((_DWORD *)v31 + 3);
                *((_DWORD *)v31 + 3) = v33 - 1;
              }
              if ( v33 == 1 )
                break;
            }
          }
          v30 += 16;
          if ( v29 == v30 )
            goto LABEL_40;
        }
        v30 += 16;
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v31 + 24LL))(v31);
      }
      while ( v29 != v30 );
LABEL_40:
      v30 = *(_QWORD *)(v25 + 8);
    }
    if ( v30 )
    {
      v4 = *(_QWORD *)(v25 + 24) - v30;
      j_j___libc_free_0(v30, v4);
    }
    v25 += 88;
  }
  v34 = a2;
  v35 = i + 88;
  if ( a2 != v55 )
  {
    v36 = i + 88;
    do
    {
      v37 = *v34;
      v38 = v34[18];
      v36 += 88;
      v34 += 22;
      *(_DWORD *)(v36 - 88) = v37;
      v39 = *((_QWORD *)v34 - 10);
      *(_DWORD *)(v36 - 16) = v38;
      v40 = *(v34 - 3);
      *(_QWORD *)(v36 - 80) = v39;
      v41 = *((_QWORD *)v34 - 9);
      *(_DWORD *)(v36 - 12) = v40;
      v42 = *(v34 - 2);
      *(_QWORD *)(v36 - 72) = v41;
      v43 = *((_QWORD *)v34 - 8);
      *(_DWORD *)(v36 - 8) = v42;
      *(_QWORD *)(v36 - 64) = v43;
      *(_QWORD *)(v36 - 56) = *((_QWORD *)v34 - 7);
      *(_DWORD *)(v36 - 48) = *(v34 - 12);
      *(_QWORD *)(v36 - 40) = *((_QWORD *)v34 - 5);
      *(_QWORD *)(v36 - 32) = *((_QWORD *)v34 - 4);
      *(_QWORD *)(v36 - 24) = *((_QWORD *)v34 - 3);
    }
    while ( v34 != v55 );
    v35 += 88
         * (((0xE8BA2E8BA2E8BA3LL * ((unsigned __int64)((char *)v34 - (char *)a2 - 88) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 1);
  }
  if ( v53 )
    j_j___libc_free_0(v53, a1[2] - v53);
  *a1 = v54;
  result = v54 + 88 * v51;
  a1[1] = v35;
  a1[2] = result;
  return result;
}
