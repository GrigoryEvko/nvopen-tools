// Function: sub_A18EF0
// Address: 0xa18ef0
//
__int64 *__fastcall sub_A18EF0(__int64 *a1, char *a2)
{
  __int64 v2; // rax
  bool v3; // zf
  __int64 v5; // rsi
  __int64 v6; // rax
  bool v7; // cf
  unsigned __int64 v8; // rax
  char *v9; // rsi
  __int64 v10; // rdx
  char *v11; // rsi
  char *v12; // r15
  __int64 i; // r12
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r13
  volatile signed __int32 *v17; // rdi
  signed __int32 v18; // eax
  signed __int32 v19; // eax
  __int64 v20; // r12
  int v21; // eax
  __int64 v23; // rbx
  char *v24; // [rsp+0h] [rbp-60h]
  __int64 v25; // [rsp+8h] [rbp-58h]
  char *v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+20h] [rbp-40h]

  v24 = (char *)a1[1];
  v2 = (__int64)&v24[-*a1] >> 5;
  v27 = (char *)*a1;
  if ( v2 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v3 = v2 == 0;
  v5 = (__int64)&v24[-*a1] >> 5;
  v6 = 1;
  if ( !v3 )
    v6 = (__int64)&v24[-*a1] >> 5;
  v7 = __CFADD__(v5, v6);
  v8 = v5 + v6;
  v9 = (char *)(a2 - v27);
  if ( v7 )
  {
    v23 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v8 )
    {
      v25 = 0;
      v10 = 32;
      v28 = 0;
      goto LABEL_7;
    }
    if ( v8 > 0x3FFFFFFFFFFFFFFLL )
      v8 = 0x3FFFFFFFFFFFFFFLL;
    v23 = 32 * v8;
  }
  v9 = (char *)(a2 - v27);
  v28 = sub_22077B0(v23);
  v10 = v28 + 32;
  v25 = v28 + v23;
LABEL_7:
  v11 = &v9[v28];
  if ( v11 )
  {
    *(_OWORD *)v11 = 0;
    *((_OWORD *)v11 + 1) = 0;
  }
  v12 = v27;
  if ( a2 != v27 )
  {
    for ( i = v28; !i; i = v14 )
    {
      v15 = *((_QWORD *)v12 + 2);
      v16 = *((_QWORD *)v12 + 1);
      if ( v15 == v16 )
      {
        v11 = (char *)(*((_QWORD *)v12 + 3) - v16);
      }
      else
      {
        do
        {
          while ( 1 )
          {
            v17 = *(volatile signed __int32 **)(v16 + 8);
            if ( v17 )
            {
              if ( &_pthread_key_create )
              {
                v18 = _InterlockedExchangeAdd(v17 + 2, 0xFFFFFFFF);
              }
              else
              {
                v18 = *((_DWORD *)v17 + 2);
                *((_DWORD *)v17 + 2) = v18 - 1;
              }
              if ( v18 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *, char *, __int64))(*(_QWORD *)v17 + 16LL))(
                  v17,
                  v11,
                  v10);
                if ( &_pthread_key_create )
                {
                  v19 = _InterlockedExchangeAdd(v17 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v19 = *((_DWORD *)v17 + 3);
                  *((_DWORD *)v17 + 3) = v19 - 1;
                }
                if ( v19 == 1 )
                  break;
              }
            }
            v16 += 16;
            if ( v15 == v16 )
              goto LABEL_26;
          }
          v16 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v17 + 24LL))(v17);
        }
        while ( v15 != v16 );
LABEL_26:
        v16 = *((_QWORD *)v12 + 1);
        v11 = (char *)(*((_QWORD *)v12 + 3) - v16);
      }
      if ( !v16 )
        goto LABEL_12;
      v12 += 32;
      j_j___libc_free_0(v16, v11);
      v14 = 32;
      if ( v12 == a2 )
      {
LABEL_29:
        v10 = i + 64;
        goto LABEL_30;
      }
LABEL_13:
      ;
    }
    *(_DWORD *)i = *(_DWORD *)v12;
    *(_QWORD *)(i + 8) = *((_QWORD *)v12 + 1);
    *(_QWORD *)(i + 16) = *((_QWORD *)v12 + 2);
    *(_QWORD *)(i + 24) = *((_QWORD *)v12 + 3);
    *((_QWORD *)v12 + 3) = 0;
    *((_QWORD *)v12 + 2) = 0;
    *((_QWORD *)v12 + 1) = 0;
LABEL_12:
    v12 += 32;
    v14 = i + 32;
    if ( v12 == a2 )
      goto LABEL_29;
    goto LABEL_13;
  }
LABEL_30:
  if ( a2 == v24 )
  {
    v20 = v10;
  }
  else
  {
    v20 = v10 + v24 - a2;
    do
    {
      v21 = *(_DWORD *)a2;
      v10 += 32;
      a2 += 32;
      *(_DWORD *)(v10 - 32) = v21;
      *(_QWORD *)(v10 - 24) = *((_QWORD *)a2 - 3);
      *(_QWORD *)(v10 - 16) = *((_QWORD *)a2 - 2);
      *(_QWORD *)(v10 - 8) = *((_QWORD *)a2 - 1);
    }
    while ( v10 != v20 );
  }
  if ( v27 )
    j_j___libc_free_0(v27, a1[2] - (_QWORD)v27);
  *a1 = v28;
  a1[1] = v20;
  a1[2] = v25;
  return a1;
}
