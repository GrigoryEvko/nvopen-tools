// Function: sub_3118D30
// Address: 0x3118d30
//
unsigned __int64 *__fastcall sub_3118D30(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        unsigned __int64 *a3,
        unsigned __int64 *a4,
        unsigned __int64 *a5,
        __int64 a6)
{
  unsigned __int64 *v6; // r14
  unsigned __int64 *v8; // rbx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r8
  __int64 v12; // r9
  size_t v13; // rcx
  size_t v14; // r8
  __int64 *v15; // rdi
  size_t v16; // rdx
  int v17; // eax
  __int64 v18; // rcx
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r13
  unsigned __int64 *v23; // r15
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // rbx
  __int64 v27; // rcx
  __int64 v28; // r14
  unsigned __int64 *v29; // r13
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // r12
  unsigned __int64 v32; // r15
  __int64 v33; // r15
  size_t v36; // [rsp+18h] [rbp-A8h]
  unsigned __int64 *v37; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v38; // [rsp+20h] [rbp-A0h]
  size_t v39; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v40; // [rsp+20h] [rbp-A0h]
  unsigned __int64 *v41; // [rsp+20h] [rbp-A0h]
  unsigned __int64 *v42; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v43; // [rsp+28h] [rbp-98h]
  __int64 *v44; // [rsp+28h] [rbp-98h]
  unsigned __int64 v45; // [rsp+28h] [rbp-98h]
  __int64 v46; // [rsp+28h] [rbp-98h]
  __int64 v47; // [rsp+28h] [rbp-98h]
  int v48; // [rsp+28h] [rbp-98h]
  int v49; // [rsp+28h] [rbp-98h]
  void *s1; // [rsp+30h] [rbp-90h] BYREF
  size_t n; // [rsp+38h] [rbp-88h]
  __int64 v52; // [rsp+40h] [rbp-80h] BYREF
  char v53; // [rsp+50h] [rbp-70h]
  void *s2; // [rsp+60h] [rbp-60h] BYREF
  size_t v55; // [rsp+68h] [rbp-58h]
  __int64 v56; // [rsp+70h] [rbp-50h] BYREF
  char v57; // [rsp+80h] [rbp-40h]

  v6 = a1;
  v8 = a3;
  if ( a1 != a2 && a3 != a4 )
  {
    do
    {
      sub_31185E0((__int64)&s2, a6, *(_DWORD *)(*v6 + 12));
      sub_31185E0((__int64)&s1, a6, *(_DWORD *)(*v8 + 12));
      v13 = n;
      v14 = v55;
      v15 = (__int64 *)s1;
      v16 = v55;
      if ( n <= v55 )
        v16 = n;
      if ( !v16
        || (v36 = v55, v39 = n, v44 = (__int64 *)s1, v17 = memcmp(s1, s2, v16), v15 = v44, v13 = v39, v14 = v36, !v17) )
      {
        v18 = v13 - v14;
        v17 = 0x7FFFFFFF;
        if ( v18 < 0x80000000LL )
        {
          v17 = 0x80000000;
          if ( v18 > (__int64)0xFFFFFFFF7FFFFFFFLL )
            v17 = v18;
        }
      }
      if ( v53 )
      {
        v53 = 0;
        if ( v15 != &v52 )
        {
          v49 = v17;
          j_j___libc_free_0((unsigned __int64)v15);
          v17 = v49;
        }
      }
      if ( v57 )
      {
        v57 = 0;
        if ( s2 != &v56 )
        {
          v48 = v17;
          j_j___libc_free_0((unsigned __int64)s2);
          v17 = v48;
        }
      }
      if ( v17 < 0 )
      {
        v10 = *v8;
        *v8 = 0;
        v11 = *a5;
        *a5 = v10;
        if ( v11 )
        {
          v12 = *(_QWORD *)(v11 + 24);
          if ( v12 )
          {
            v38 = v11;
            v43 = *(_QWORD *)(v11 + 24);
            sub_C7D6A0(*(_QWORD *)(v12 + 8), 16LL * *(unsigned int *)(v12 + 24), 8);
            j_j___libc_free_0(v43);
            v11 = v38;
          }
          j_j___libc_free_0(v11);
        }
        ++v8;
        ++a5;
        if ( v6 == a2 )
          break;
      }
      else
      {
        v19 = *v6;
        *v6 = 0;
        v20 = *a5;
        *a5 = v19;
        if ( v20 )
        {
          v21 = *(_QWORD *)(v20 + 24);
          if ( v21 )
          {
            v40 = v20;
            v45 = *(_QWORD *)(v20 + 24);
            sub_C7D6A0(*(_QWORD *)(v21 + 8), 16LL * *(unsigned int *)(v21 + 24), 8);
            j_j___libc_free_0(v45);
            v20 = v40;
          }
          j_j___libc_free_0(v20);
        }
        ++v6;
        ++a5;
        if ( v6 == a2 )
          break;
      }
    }
    while ( v8 != a4 );
  }
  v46 = (char *)a2 - (char *)v6;
  v22 = a2 - v6;
  if ( (char *)a2 - (char *)v6 > 0 )
  {
    v41 = v8;
    v23 = a5;
    v37 = a5;
    do
    {
      v24 = *v6;
      *v6 = 0;
      v25 = *v23;
      *v23 = v24;
      if ( v25 )
      {
        v26 = *(_QWORD *)(v25 + 24);
        if ( v26 )
        {
          sub_C7D6A0(*(_QWORD *)(v26 + 8), 16LL * *(unsigned int *)(v26 + 24), 8);
          j_j___libc_free_0(v26);
        }
        j_j___libc_free_0(v25);
      }
      ++v6;
      ++v23;
      --v22;
    }
    while ( v22 );
    v27 = v46;
    v8 = v41;
    if ( v46 <= 0 )
      v27 = 8;
    a5 = (unsigned __int64 *)((char *)v37 + v27);
  }
  v28 = a4 - v8;
  if ( (char *)a4 - (char *)v8 > 0 )
  {
    v47 = (char *)a4 - (char *)v8;
    v29 = a5;
    v42 = a5;
    do
    {
      v30 = *v8;
      *v8 = 0;
      v31 = *v29;
      *v29 = v30;
      if ( v31 )
      {
        v32 = *(_QWORD *)(v31 + 24);
        if ( v32 )
        {
          sub_C7D6A0(*(_QWORD *)(v32 + 8), 16LL * *(unsigned int *)(v32 + 24), 8);
          j_j___libc_free_0(v32);
        }
        j_j___libc_free_0(v31);
      }
      ++v8;
      ++v29;
      --v28;
    }
    while ( v28 );
    v33 = v47;
    if ( v47 <= 0 )
      v33 = 8;
    return (unsigned __int64 *)((char *)v42 + v33);
  }
  return a5;
}
