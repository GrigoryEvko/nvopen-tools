// Function: sub_36FD240
// Address: 0x36fd240
//
__int64 *__fastcall sub_36FD240(__int64 *a1, __int64 **a2)
{
  __int64 *v2; // r8
  __int64 v3; // rax
  __int64 v4; // r13
  void *v5; // rax
  __int64 v6; // r12
  unsigned __int64 v7; // r14
  unsigned int v8; // edx
  __int64 v9; // rax
  __int64 *v10; // rbx
  char *v11; // r13
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rdx
  __int64 *v16; // rax
  __int64 v17; // rdx
  char *v18; // r14
  unsigned __int64 v19; // rax
  char *v20; // r12
  size_t *v21; // r15
  size_t **v22; // rbx
  const void *v23; // rdi
  size_t *v24; // r14
  size_t v25; // rdx
  size_t v26; // rax
  int v27; // esi
  bool v28; // cc
  int v29; // eax
  size_t v30; // r9
  size_t v31; // rcx
  size_t v32; // rdx
  int v33; // eax
  __int64 *v35; // [rsp+0h] [rbp-60h]
  size_t v36; // [rsp+8h] [rbp-58h]
  size_t v37; // [rsp+10h] [rbp-50h]
  __int64 *v38; // [rsp+18h] [rbp-48h]
  __int64 *v39; // [rsp+18h] [rbp-48h]
  __int64 *v40; // [rsp+18h] [rbp-48h]
  __int64 v41[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = a1;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v3 = *((unsigned int *)a2 + 3);
  if ( *((_DWORD *)a2 + 3) )
  {
    v4 = 8 * v3;
    v5 = (void *)sub_22077B0(8 * v3);
    v2 = a1;
    v6 = (__int64)v5;
    v7 = *a1;
    if ( v2[1] - *v2 > 0 )
    {
      memmove(v5, (const void *)*a1, a1[1] - *a1);
      v2 = a1;
    }
    else if ( !v7 )
    {
LABEL_4:
      *v2 = v6;
      v2[1] = v6;
      v2[2] = v4 + v6;
      goto LABEL_5;
    }
    v39 = v2;
    j_j___libc_free_0(v7);
    v2 = v39;
    goto LABEL_4;
  }
LABEL_5:
  v8 = *((_DWORD *)a2 + 2);
  if ( v8 )
  {
    v9 = **a2;
    v10 = *a2;
    if ( v9 )
      goto LABEL_8;
    do
    {
      do
      {
        v9 = v10[1];
        ++v10;
      }
      while ( !v9 );
LABEL_8:
      ;
    }
    while ( v9 == -8 );
    v11 = (char *)v2[1];
    v12 = (__int64)&(*a2)[v8];
    while ( (__int64 *)v12 != v10 )
    {
      while ( 1 )
      {
        v13 = *v10;
        v41[0] = *v10;
        if ( v11 == (char *)v2[2] )
        {
          v40 = v2;
          sub_36FD0B0((__int64)v2, v11, v41);
          v2 = v40;
          v11 = (char *)v40[1];
        }
        else
        {
          if ( v11 )
          {
            *(_QWORD *)v11 = v13;
            v11 = (char *)v2[1];
          }
          v11 += 8;
          v2[1] = (__int64)v11;
        }
        v14 = v10[1];
        v15 = v10 + 1;
        if ( v14 == -8 || !v14 )
          break;
        ++v10;
        if ( (_QWORD *)v12 == v15 )
          goto LABEL_20;
      }
      v16 = v10 + 2;
      do
      {
        do
        {
          v17 = *v16;
          v10 = v16++;
        }
        while ( v17 == -8 );
      }
      while ( !v17 );
    }
  }
  else
  {
    v11 = (char *)v2[1];
  }
LABEL_20:
  v18 = (char *)*v2;
  if ( (char *)*v2 != v11 )
  {
    v38 = v2;
    _BitScanReverse64(&v19, (v11 - v18) >> 3);
    sub_36FC430(*v2, v11, 2LL * (int)(63 - (v19 ^ 0x3F)));
    if ( v11 - v18 <= 128 )
    {
      sub_36FC280(v18, v11);
      return v38;
    }
    v20 = v18 + 128;
    sub_36FC280(v18, v18 + 128);
    v2 = v38;
    if ( v18 + 128 != v11 )
    {
LABEL_23:
      v21 = *(size_t **)v20;
      v22 = (size_t **)v20;
      v23 = (const void *)(*(_QWORD *)v20 + 16LL);
      while ( 1 )
      {
        v24 = *(v22 - 1);
        v25 = v21[1];
        v26 = v24[1];
        v27 = *(_DWORD *)(v26 + 80);
        v28 = *(_DWORD *)(v25 + 80) <= v27;
        if ( *(_DWORD *)(v25 + 80) == v27
          && (v29 = *(_DWORD *)(v26 + 84), v28 = *(_DWORD *)(v25 + 84) <= v29, *(_DWORD *)(v25 + 84) == v29) )
        {
          v30 = *v24;
          v31 = *v21;
          v32 = *v21;
          if ( *v24 <= *v21 )
            v32 = *v24;
          if ( v32
            && (v35 = v2, v36 = *v21, v37 = *v24, v33 = memcmp(v23, v24 + 2, v32), v30 = v37, v31 = v36, v2 = v35, v33) )
          {
            if ( v33 >= 0 )
              goto LABEL_34;
          }
          else if ( v30 == v31 || v30 <= v31 )
          {
LABEL_34:
            v20 += 8;
            *v22 = v21;
            if ( v11 == v20 )
              return v2;
            goto LABEL_23;
          }
        }
        else if ( v28 )
        {
          goto LABEL_34;
        }
        *v22-- = v24;
      }
    }
  }
  return v2;
}
