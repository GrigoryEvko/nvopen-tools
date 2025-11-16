// Function: sub_A18BD0
// Address: 0xa18bd0
//
__int64 *__fastcall sub_A18BD0(__int64 *a1, char *a2, int *a3, _QWORD *a4)
{
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  bool v7; // cf
  unsigned __int64 v8; // rax
  char *v9; // rdx
  __int64 v10; // rbx
  char *v11; // rax
  __int64 v12; // rdx
  int v13; // ecx
  char *v14; // r14
  __int64 i; // r15
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r13
  volatile signed __int32 *v19; // rbx
  signed __int32 v20; // eax
  signed __int32 v21; // eax
  char *v22; // rax
  __int64 v23; // rdx
  int v24; // ecx
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rsi
  _QWORD *v29; // [rsp+8h] [rbp-78h]
  __int64 v30; // [rsp+20h] [rbp-60h]
  char *v31; // [rsp+30h] [rbp-50h]
  __int64 v32; // [rsp+38h] [rbp-48h]
  char *v33; // [rsp+40h] [rbp-40h]

  v33 = (char *)a1[1];
  v31 = (char *)*a1;
  v5 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)&v33[-*a1] >> 3);
  if ( v5 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)&v33[-*a1] >> 3);
  v7 = __CFADD__(v6, v5);
  v8 = v6 - 0x3333333333333333LL * ((__int64)&v33[-*a1] >> 3);
  v9 = (char *)(a2 - v31);
  if ( v7 )
  {
    v26 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v8 )
    {
      v30 = 0;
      v10 = 40;
      v32 = 0;
      goto LABEL_7;
    }
    if ( v8 > 0x333333333333333LL )
      v8 = 0x333333333333333LL;
    v26 = 40 * v8;
  }
  v29 = a4;
  v27 = sub_22077B0(v26);
  v9 = (char *)(a2 - v31);
  v32 = v27;
  a4 = v29;
  v30 = v27 + v26;
  v10 = v27 + 40;
LABEL_7:
  v11 = &v9[v32];
  if ( &v9[v32] )
  {
    v12 = *a4;
    v13 = *a3;
    *((_QWORD *)v11 + 2) = 0;
    *((_QWORD *)v11 + 3) = 0;
    *(_DWORD *)v11 = v13;
    *((_QWORD *)v11 + 1) = v12;
    *((_QWORD *)v11 + 4) = 0;
  }
  v14 = v31;
  if ( a2 != v31 )
  {
    for ( i = v32; !i; i = v16 )
    {
      v17 = *((_QWORD *)v14 + 3);
      v18 = *((_QWORD *)v14 + 2);
      if ( v17 == v18 )
      {
        v28 = *((_QWORD *)v14 + 4) - v18;
      }
      else
      {
        do
        {
          while ( 1 )
          {
            v19 = *(volatile signed __int32 **)(v18 + 8);
            if ( v19 )
            {
              if ( &_pthread_key_create )
              {
                v20 = _InterlockedExchangeAdd(v19 + 2, 0xFFFFFFFF);
              }
              else
              {
                v20 = *((_DWORD *)v19 + 2);
                *((_DWORD *)v19 + 2) = v20 - 1;
              }
              if ( v20 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 16LL))(v19);
                if ( &_pthread_key_create )
                {
                  v21 = _InterlockedExchangeAdd(v19 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v21 = *((_DWORD *)v19 + 3);
                  *((_DWORD *)v19 + 3) = v21 - 1;
                }
                if ( v21 == 1 )
                  break;
              }
            }
            v18 += 16;
            if ( v17 == v18 )
              goto LABEL_26;
          }
          v18 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 24LL))(v19);
        }
        while ( v17 != v18 );
LABEL_26:
        v18 = *((_QWORD *)v14 + 2);
        v28 = *((_QWORD *)v14 + 4) - v18;
      }
      if ( !v18 )
        goto LABEL_12;
      v14 += 40;
      j_j___libc_free_0(v18, v28);
      v16 = 40;
      if ( v14 == a2 )
      {
LABEL_29:
        v10 = i + 80;
        goto LABEL_30;
      }
LABEL_13:
      ;
    }
    *(_DWORD *)i = *(_DWORD *)v14;
    *(_QWORD *)(i + 8) = *((_QWORD *)v14 + 1);
    *(_QWORD *)(i + 16) = *((_QWORD *)v14 + 2);
    *(_QWORD *)(i + 24) = *((_QWORD *)v14 + 3);
    *(_QWORD *)(i + 32) = *((_QWORD *)v14 + 4);
    *((_QWORD *)v14 + 4) = 0;
    *((_QWORD *)v14 + 3) = 0;
    *((_QWORD *)v14 + 2) = 0;
LABEL_12:
    v14 += 40;
    v16 = i + 40;
    if ( v14 == a2 )
      goto LABEL_29;
    goto LABEL_13;
  }
LABEL_30:
  v22 = a2;
  if ( a2 != v33 )
  {
    v23 = v10;
    do
    {
      v24 = *(_DWORD *)v22;
      v23 += 40;
      v22 += 40;
      *(_DWORD *)(v23 - 40) = v24;
      *(_QWORD *)(v23 - 32) = *((_QWORD *)v22 - 4);
      *(_QWORD *)(v23 - 24) = *((_QWORD *)v22 - 3);
      *(_QWORD *)(v23 - 16) = *((_QWORD *)v22 - 2);
      *(_QWORD *)(v23 - 8) = *((_QWORD *)v22 - 1);
    }
    while ( v22 != v33 );
    v10 += 8 * ((unsigned __int64)(v22 - a2 - 40) >> 3) + 40;
  }
  if ( v31 )
    j_j___libc_free_0(v31, a1[2] - (_QWORD)v31);
  *a1 = v32;
  a1[1] = v10;
  a1[2] = v30;
  return a1;
}
