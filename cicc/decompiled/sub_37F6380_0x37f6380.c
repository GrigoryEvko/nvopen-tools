// Function: sub_37F6380
// Address: 0x37f6380
//
void __fastcall sub_37F6380(__int64 **a1, unsigned __int64 a2)
{
  __int64 **v2; // r13
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r12
  __int64 *v5; // r14
  __int64 v6; // rdx
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // rdx
  __int64 *v9; // rax
  __int64 v10; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  _QWORD *v13; // rax
  unsigned __int64 v14; // rdx
  __int64 *v15; // rbx
  __int64 *v16; // r12
  __int64 v17; // rax
  unsigned __int64 v18; // r13
  void **v19; // rax
  void **v20; // r9
  void *v21; // rdi
  unsigned int v22; // r11d
  __int64 *v23; // r14
  __int64 *v24; // rbx
  __int64 v25; // rax
  unsigned __int64 *v26; // rax
  unsigned __int64 v27; // r12
  size_t v28; // rdx
  unsigned __int64 v29; // rcx
  __int64 v30; // rax
  unsigned int v31; // [rsp-64h] [rbp-64h]
  unsigned int v32; // [rsp-64h] [rbp-64h]
  void **v33; // [rsp-60h] [rbp-60h]
  void **v34; // [rsp-60h] [rbp-60h]
  __int64 v36; // [rsp-50h] [rbp-50h]
  unsigned __int64 v37; // [rsp-48h] [rbp-48h]
  unsigned __int64 v38; // [rsp-48h] [rbp-48h]
  __int64 *v39; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v2 = a1;
  v3 = a2;
  v4 = (unsigned __int64)a1[1];
  v5 = *a1;
  v6 = v4 - (_QWORD)*a1;
  v7 = v6 >> 3;
  if ( a2 <= (__int64)((__int64)a1[2] - v4) >> 3 )
  {
    v8 = a2;
    v9 = a1[1];
    do
    {
      if ( v9 )
        *v9 = 0;
      ++v9;
      --v8;
    }
    while ( v8 );
    a1[1] = (__int64 *)(v4 + 8 * a2);
    return;
  }
  if ( 0xFFFFFFFFFFFFFFFLL - v7 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v10 = a1[1] - *a1;
  if ( a2 >= v7 )
    v10 = a2;
  v11 = __CFADD__(v7, v10);
  v12 = v7 + v10;
  if ( v11 )
  {
    v29 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v12 )
    {
      v37 = 0;
      v39 = 0;
      goto LABEL_15;
    }
    if ( v12 > 0xFFFFFFFFFFFFFFFLL )
      v12 = 0xFFFFFFFFFFFFFFFLL;
    v29 = 8 * v12;
  }
  v36 = (char *)a1[1] - (char *)*a1;
  v38 = v29;
  v30 = sub_22077B0(v29);
  v4 = (unsigned __int64)a1[1];
  v5 = *a1;
  v6 = v36;
  v39 = (__int64 *)v30;
  v37 = v30 + v38;
LABEL_15:
  v13 = (__int64 *)((char *)v39 + v6);
  v14 = a2;
  do
  {
    if ( v13 )
      *v13 = 0;
    ++v13;
    --v14;
  }
  while ( v14 );
  if ( v5 != (__int64 *)v4 )
  {
    v15 = (__int64 *)v4;
    v16 = v39;
    do
    {
      if ( v16 )
      {
        v17 = *v5;
        *v16 = *v5;
        if ( v17 )
        {
          if ( (v17 & 1) != 0 )
          {
            v18 = v17 & 0xFFFFFFFFFFFFFFFELL;
            if ( (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            {
              v19 = (void **)sub_22077B0(0x30u);
              v20 = v19;
              if ( v19 )
              {
                v21 = v19 + 2;
                *v19 = v19 + 2;
                v19[1] = (void *)0x400000000LL;
                v22 = *(_DWORD *)(v18 + 8);
                if ( v22 )
                {
                  if ( v19 != (void **)v18 )
                  {
                    v28 = 8LL * v22;
                    if ( v22 <= 4
                      || (v32 = *(_DWORD *)(v18 + 8),
                          v34 = v19,
                          sub_C8D5F0((__int64)v19, v21, v22, 8u, v22, (__int64)v19),
                          v20 = v34,
                          v22 = v32,
                          v28 = 8LL * *(unsigned int *)(v18 + 8),
                          v21 = *v34,
                          v28) )
                    {
                      v31 = v22;
                      v33 = v20;
                      memcpy(v21, *(const void **)v18, v28);
                      v22 = v31;
                      v20 = v33;
                    }
                    *((_DWORD *)v20 + 2) = v22;
                  }
                }
              }
              *v16 = (unsigned __int64)v20 | 1;
            }
          }
        }
      }
      ++v5;
      ++v16;
    }
    while ( v5 != v15 );
    v2 = a1;
    v3 = a2;
    v23 = a1[1];
    v4 = (unsigned __int64)*a1;
    if ( v23 != *a1 )
    {
      v24 = *a1;
      do
      {
        v25 = *v24;
        if ( *v24 )
        {
          if ( (v25 & 1) != 0 )
          {
            v26 = (unsigned __int64 *)(v25 & 0xFFFFFFFFFFFFFFFELL);
            v27 = (unsigned __int64)v26;
            if ( v26 )
            {
              if ( (unsigned __int64 *)*v26 != v26 + 2 )
                _libc_free(*v26);
              j_j___libc_free_0(v27);
            }
          }
        }
        ++v24;
      }
      while ( v23 != v24 );
      v3 = a2;
      v4 = (unsigned __int64)*a1;
    }
  }
  if ( v4 )
    j_j___libc_free_0(v4);
  *v2 = v39;
  v2[1] = &v39[v7 + v3];
  v2[2] = (__int64 *)v37;
}
