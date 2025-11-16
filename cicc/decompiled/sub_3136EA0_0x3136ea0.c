// Function: sub_3136EA0
// Address: 0x3136ea0
//
__int64 __fastcall sub_3136EA0(
        __int64 a1,
        const void *a2,
        size_t a3,
        const void *a4,
        size_t a5,
        __int64 a6,
        unsigned int a7,
        _DWORD *a8)
{
  const void *v8; // r10
  unsigned int v9; // r15d
  size_t v11; // rax
  size_t v12; // rdi
  size_t v13; // rbx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rcx
  int v17; // r8d
  size_t v18; // r8
  size_t v19; // rdi
  void *v20; // r9
  size_t v21; // rdi
  void *v22; // r8
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // rcx
  int v26; // r8d
  __int64 v27; // r8
  __int64 v28; // r9
  size_t v29; // r12
  size_t v30; // rdi
  void *v31; // r14
  void *v32; // r8
  size_t v33; // rdi
  size_t v34; // rax
  unsigned __int64 v35; // rdx
  __int64 v36; // r12
  char *p_dest; // rdi
  size_t v39; // [rsp+0h] [rbp-110h]
  size_t v40; // [rsp+0h] [rbp-110h]
  size_t v41; // [rsp+8h] [rbp-108h]
  __int64 v42; // [rsp+8h] [rbp-108h]
  void *v44; // [rsp+8h] [rbp-108h]
  void *src; // [rsp+20h] [rbp-F0h] BYREF
  size_t n; // [rsp+28h] [rbp-E8h]
  _QWORD v48[2]; // [rsp+30h] [rbp-E0h] BYREF
  char *v49; // [rsp+40h] [rbp-D0h] BYREF
  size_t v50; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v51; // [rsp+50h] [rbp-C0h]
  char v52; // [rsp+58h] [rbp-B8h] BYREF
  char dest; // [rsp+59h] [rbp-B7h] BYREF

  v8 = a4;
  v9 = a6;
  v49 = &v52;
  v51 = 128;
  v52 = 59;
  v50 = 1;
  if ( a5 + 1 > 0x80 )
  {
    v39 = a5;
    sub_C8D290((__int64)&v49, &v52, a5 + 1, 1u, a5, a6);
    v8 = a4;
    a5 = v39;
    p_dest = &v49[v50];
  }
  else
  {
    v11 = 1;
    if ( !a5 )
      goto LABEL_3;
    p_dest = &dest;
  }
  v42 = a5;
  memcpy(p_dest, v8, a5);
  a5 = v42;
  v11 = v42 + v50;
  v50 = v11;
  if ( v51 < v11 + 1 )
  {
    sub_C8D290((__int64)&v49, &v52, v11 + 1, 1u, v42, a6);
    v11 = v50;
  }
LABEL_3:
  v49[v11] = 59;
  v12 = v50 + 1;
  v50 = v12;
  if ( a3 + v12 > v51 )
  {
    sub_C8D290((__int64)&v49, &v52, a3 + v12, 1u, a5, a6);
    v12 = v50;
  }
  if ( a3 )
  {
    memcpy(&v49[v12], a2, a3);
    v12 = v50;
  }
  v13 = v12 + a3;
  v50 = v13;
  if ( v13 + 1 > v51 )
  {
    sub_C8D290((__int64)&v49, &v52, v13 + 1, 1u, a5, a6);
    v13 = v50;
  }
  v49[v13] = 59;
  ++v50;
  if ( v9 <= 9 )
  {
    v15 = 1;
  }
  else if ( v9 <= 0x63 )
  {
    v15 = 2;
  }
  else if ( v9 <= 0x3E7 )
  {
    v15 = 3;
  }
  else
  {
    v14 = v9;
    if ( v9 <= 0x270F )
    {
      v15 = 4;
    }
    else
    {
      LODWORD(v15) = 1;
      while ( 1 )
      {
        v16 = v14;
        v17 = v15;
        v15 = (unsigned int)(v15 + 4);
        v14 /= 0x2710u;
        if ( v16 <= 0x1869F )
          break;
        if ( (unsigned int)v14 <= 0x63 )
        {
          v15 = (unsigned int)(v17 + 5);
          break;
        }
        if ( (unsigned int)v14 <= 0x3E7 )
        {
          v15 = (unsigned int)(v17 + 6);
          break;
        }
        if ( (unsigned int)v14 <= 0x270F )
        {
          v15 = (unsigned int)(v17 + 7);
          break;
        }
      }
    }
  }
  src = v48;
  sub_2240A50((__int64 *)&src, v15, 0);
  sub_2554A60(src, n, v9);
  v18 = n;
  v19 = v50;
  v20 = src;
  if ( n + v50 > v51 )
  {
    v40 = n;
    v44 = src;
    sub_C8D290((__int64)&v49, &v52, n + v50, 1u, n, (__int64)src);
    v19 = v50;
    v18 = v40;
    v20 = v44;
  }
  if ( v18 )
  {
    v41 = v18;
    memcpy(&v49[v19], v20, v18);
    v19 = v50;
    v18 = v41;
  }
  v21 = v18 + v19;
  v22 = src;
  v50 = v21;
  if ( src != v48 )
  {
    j_j___libc_free_0((unsigned __int64)src);
    v21 = v50;
  }
  if ( v21 + 1 > v51 )
  {
    sub_C8D290((__int64)&v49, &v52, v21 + 1, 1u, (__int64)v22, (__int64)v20);
    v21 = v50;
  }
  v49[v21] = 59;
  ++v50;
  if ( a7 <= 9 )
  {
    v24 = 1;
  }
  else if ( a7 <= 0x63 )
  {
    v24 = 2;
  }
  else if ( a7 <= 0x3E7 )
  {
    v24 = 3;
  }
  else
  {
    v23 = a7;
    if ( a7 <= 0x270F )
    {
      v24 = 4;
    }
    else
    {
      LODWORD(v24) = 1;
      while ( 1 )
      {
        v25 = v23;
        v26 = v24;
        v24 = (unsigned int)(v24 + 4);
        v23 /= 0x2710u;
        if ( v25 <= 0x1869F )
          break;
        if ( (unsigned int)v23 <= 0x63 )
        {
          v24 = (unsigned int)(v26 + 5);
          break;
        }
        if ( (unsigned int)v23 <= 0x3E7 )
        {
          v24 = (unsigned int)(v26 + 6);
          break;
        }
        if ( (unsigned int)v23 <= 0x270F )
        {
          v24 = (unsigned int)(v26 + 7);
          break;
        }
      }
    }
  }
  src = v48;
  sub_2240A50((__int64 *)&src, v24, 0);
  sub_2554A60(src, n, a7);
  v29 = n;
  v30 = v50;
  v31 = src;
  if ( n + v50 > v51 )
  {
    sub_C8D290((__int64)&v49, &v52, n + v50, 1u, v27, v28);
    v30 = v50;
  }
  if ( v29 )
  {
    memcpy(&v49[v30], v31, v29);
    v30 = v50;
  }
  v32 = src;
  v33 = v29 + v30;
  v50 = v33;
  if ( src != v48 )
  {
    j_j___libc_free_0((unsigned __int64)src);
    v33 = v50;
  }
  if ( v33 + 1 > v51 )
  {
    sub_C8D290((__int64)&v49, &v52, v33 + 1, 1u, (__int64)v32, v28);
    v33 = v50;
  }
  v49[v33] = 59;
  v34 = v50 + 1;
  v35 = v50 + 2;
  ++v50;
  if ( v35 > v51 )
  {
    sub_C8D290((__int64)&v49, &v52, v35, 1u, (__int64)v32, v28);
    v34 = v50;
  }
  v49[v34] = 59;
  v36 = sub_3135B50(a1, v49, ++v50, a8);
  if ( v49 != &v52 )
    _libc_free((unsigned __int64)v49);
  return v36;
}
