// Function: sub_1BC2E70
// Address: 0x1bc2e70
//
__int64 *__fastcall sub_1BC2E70(__int64 *a1, char *a2, _QWORD *a3)
{
  __int64 v3; // rax
  bool v4; // zf
  __int64 v6; // rdx
  __int64 v7; // rax
  bool v8; // cf
  unsigned __int64 v9; // rax
  __int64 v10; // rbx
  _QWORD *v11; // r13
  __int64 v12; // rax
  char *v13; // rbx
  __int64 v14; // r13
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 *v19; // r12
  __int64 *v20; // r15
  __int64 v21; // rbx
  __int64 v22; // r14
  unsigned __int64 v23; // rdi
  char *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v28; // rbx
  char *v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+18h] [rbp-68h]
  char *v32; // [rsp+28h] [rbp-58h]
  __int64 v33; // [rsp+30h] [rbp-50h]
  char *v34; // [rsp+38h] [rbp-48h]
  _QWORD *i; // [rsp+48h] [rbp-38h]

  v29 = (char *)a1[1];
  v3 = (__int64)&v29[-*a1] >> 4;
  v32 = (char *)*a1;
  if ( v3 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v4 = v3 == 0;
  v6 = (__int64)&v29[-*a1] >> 4;
  v7 = 1;
  if ( !v4 )
    v7 = (__int64)&v29[-*a1] >> 4;
  v8 = __CFADD__(v6, v7);
  v9 = v6 + v7;
  if ( v8 )
  {
    v28 = 0x7FFFFFFFFFFFFFF0LL;
  }
  else
  {
    if ( !v9 )
    {
      v30 = 0;
      v10 = 16;
      v33 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x7FFFFFFFFFFFFFFLL )
      v9 = 0x7FFFFFFFFFFFFFFLL;
    v28 = 16 * v9;
  }
  v33 = sub_22077B0(v28);
  v30 = v33 + v28;
  v10 = v33 + 16;
LABEL_7:
  v11 = (_QWORD *)(v33 + a2 - v32);
  if ( v11 )
  {
    *v11 = *a3;
    v12 = a3[1];
    a3[1] = 0;
    v11[1] = v12;
  }
  if ( a2 != v32 )
  {
    v13 = v32;
    for ( i = (_QWORD *)v33; ; i += 2 )
    {
      if ( i )
      {
        *i = *(_QWORD *)v13;
        i[1] = *((_QWORD *)v13 + 1);
        *((_QWORD *)v13 + 1) = 0;
      }
      else
      {
        v14 = *((_QWORD *)v13 + 1);
        if ( v14 )
        {
          v15 = *(_QWORD *)(v14 + 104);
          if ( v15 != v14 + 120 )
            _libc_free(v15);
          v16 = *(unsigned int *)(v14 + 96);
          if ( (_DWORD)v16 )
          {
            v17 = *(_QWORD *)(v14 + 80);
            v18 = v17 + 88 * v16;
            do
            {
              if ( *(_QWORD *)v17 != -8 && *(_QWORD *)v17 != -16 && (*(_BYTE *)(v17 + 16) & 1) == 0 )
                j___libc_free_0(*(_QWORD *)(v17 + 24));
              v17 += 88;
            }
            while ( v18 != v17 );
          }
          j___libc_free_0(*(_QWORD *)(v14 + 80));
          j___libc_free_0(*(_QWORD *)(v14 + 48));
          v19 = *(__int64 **)(v14 + 8);
          if ( *(__int64 **)(v14 + 16) != v19 )
          {
            v34 = v13;
            v20 = *(__int64 **)(v14 + 16);
            do
            {
              v21 = *v19;
              if ( *v19 )
              {
                v22 = v21 + 112LL * *(_QWORD *)(v21 - 8);
                while ( v21 != v22 )
                {
                  v22 -= 112;
                  v23 = *(_QWORD *)(v22 + 32);
                  if ( v23 != v22 + 48 )
                    _libc_free(v23);
                }
                j_j_j___libc_free_0_0(v21 - 8);
              }
              ++v19;
            }
            while ( v20 != v19 );
            v13 = v34;
            v19 = *(__int64 **)(v14 + 8);
          }
          if ( v19 )
            j_j___libc_free_0(v19, *(_QWORD *)(v14 + 24) - (_QWORD)v19);
          j_j___libc_free_0(v14, 232);
        }
      }
      v13 += 16;
      if ( v13 == a2 )
        break;
    }
    v10 = (__int64)(i + 4);
  }
  if ( a2 != v29 )
  {
    v24 = a2;
    v25 = v10;
    do
    {
      v26 = *(_QWORD *)v24;
      v24 += 16;
      v25 += 16;
      *(_QWORD *)(v25 - 16) = v26;
      *(_QWORD *)(v25 - 8) = *((_QWORD *)v24 - 1);
    }
    while ( v24 != v29 );
    v10 += v29 - a2;
  }
  if ( v32 )
    j_j___libc_free_0(v32, a1[2] - (_QWORD)v32);
  *a1 = v33;
  a1[1] = v10;
  a1[2] = v30;
  return a1;
}
