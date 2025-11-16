// Function: sub_16F3FB0
// Address: 0x16f3fb0
//
char __fastcall sub_16F3FB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v5; // r12
  __int64 v6; // rcx
  __int64 *v7; // r8
  __int64 **v8; // r15
  __int64 **v9; // rbx
  unsigned __int8 **v10; // r14
  __int64 *v11; // rax
  __int64 *v12; // r12
  __int64 v13; // rax
  size_t v14; // r15
  __int64 v15; // rbx
  __int64 v16; // rcx
  int v17; // r8d
  _QWORD *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r14
  __int64 v21; // r13
  __int64 v22; // r12
  char *v23; // rdi
  __int64 v24; // r8
  __int64 v25; // rax
  _QWORD *v26; // rdx
  _QWORD *v27; // r8
  _QWORD *v28; // r8
  unsigned __int8 **v30; // [rsp+10h] [rbp-E0h]
  __int64 *v31; // [rsp+18h] [rbp-D8h]
  unsigned __int8 **v32; // [rsp+18h] [rbp-D8h]
  __int64 v34; // [rsp+20h] [rbp-D0h]
  __int64 v35; // [rsp+28h] [rbp-C8h]
  size_t n; // [rsp+30h] [rbp-C0h]
  size_t na; // [rsp+30h] [rbp-C0h]
  __int64 v39; // [rsp+38h] [rbp-B8h]
  _QWORD *v40; // [rsp+38h] [rbp-B8h]
  __int64 v41; // [rsp+38h] [rbp-B8h]
  _QWORD *v42; // [rsp+38h] [rbp-B8h]
  __int64 v43; // [rsp+38h] [rbp-B8h]
  unsigned __int8 **v44; // [rsp+80h] [rbp-70h] BYREF
  __int64 v45; // [rsp+88h] [rbp-68h]
  size_t v46; // [rsp+90h] [rbp-60h]
  __int64 v47[2]; // [rsp+A0h] [rbp-50h] BYREF
  _QWORD v48[8]; // [rsp+B0h] [rbp-40h] BYREF

  v5 = a2;
  *(_QWORD *)(a1 + 16) = 0;
  if ( !(unsigned __int8)sub_16F23B0((unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0, 0, a4, a5) )
  {
    sub_16F2420(v47, (unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0);
    sub_16F25B0(&v44, (__int64)v47);
    v10 = v44;
    v39 = v45;
    n = v46;
    if ( (_QWORD *)v47[0] != v48 )
      j_j___libc_free_0(v47[0], v48[0] + 1LL);
    v8 = *(__int64 ***)(a1 + 8);
    v9 = &v8[8 * (unsigned __int64)*(unsigned int *)(a1 + 24)];
    if ( v8 == v9 )
    {
LABEL_16:
      if ( v10 )
      {
        if ( *v10 != (unsigned __int8 *)(v10 + 2) )
          j_j___libc_free_0(*v10, v10[2] + 1);
        j_j___libc_free_0(v10, 32);
      }
      goto LABEL_20;
    }
    v35 = a2;
    while ( 1 )
    {
LABEL_12:
      while ( !v8 )
      {
LABEL_11:
        v8 += 8;
        if ( v8 == v9 )
          goto LABEL_15;
      }
      *v8 = 0;
      v8[1] = 0;
      v8[2] = 0;
      if ( v10 )
      {
        v11 = (__int64 *)sub_22077B0(32);
        v12 = v11;
        if ( v11 )
        {
          *v11 = (__int64)(v11 + 2);
          sub_16F1520(v11, *v10, (__int64)&v10[1][(_QWORD)*v10]);
        }
        v7 = *v8;
        *v8 = v12;
        if ( v7 )
        {
          if ( (__int64 *)*v7 != v7 + 2 )
          {
            v31 = v7;
            j_j___libc_free_0(*v7, v7[2] + 1);
            v7 = v31;
          }
          j_j___libc_free_0(v7, 32);
          v12 = *v8;
        }
        v13 = v12[1];
        v8[1] = (__int64 *)*v12;
        v8[2] = (__int64 *)v13;
        goto LABEL_11;
      }
      v8 += 8;
      *(v8 - 7) = (__int64 *)v39;
      *(v8 - 6) = (__int64 *)n;
      if ( v8 == v9 )
      {
LABEL_15:
        v5 = v35;
        goto LABEL_16;
      }
    }
  }
  v8 = *(__int64 ***)(a1 + 8);
  v9 = &v8[8 * (unsigned __int64)*(unsigned int *)(a1 + 24)];
  if ( v9 != v8 )
  {
    v39 = -1;
    v10 = 0;
    n = 0;
    v35 = a2;
    goto LABEL_12;
  }
LABEL_20:
  v14 = 0;
  v15 = -1;
  v32 = 0;
  if ( !(unsigned __int8)sub_16F23B0((unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0, 0, v6, (int)v7) )
  {
    sub_16F2420(v47, (unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0);
    sub_16F25B0(&v44, (__int64)v47);
    v15 = v45;
    v14 = v46;
    v32 = v44;
    if ( (_QWORD *)v47[0] != v48 )
      j_j___libc_free_0(v47[0], v48[0] + 1LL);
  }
  LOBYTE(v18) = sub_16F23B0((unsigned __int8 *)0xFFFFFFFFFFFFFFFELL, 0, 0, v16, v17);
  if ( (_BYTE)v18 )
  {
    v30 = 0;
    v20 = -2;
    na = 0;
    if ( v5 != a3 )
    {
LABEL_23:
      v21 = v5;
      v22 = a3;
      while ( 1 )
      {
        v23 = *(char **)(v21 + 8);
        v24 = *(_QWORD *)(v21 + 16);
        LOBYTE(v18) = v23 + 1 == 0;
        if ( v15 == -1 || (LOBYTE(v18) = v23 + 2 == 0, v15 == -2) )
        {
          if ( (_BYTE)v18 )
            goto LABEL_35;
        }
        else if ( v24 == v14 )
        {
          v43 = *(_QWORD *)(v21 + 16);
          if ( !v14 )
            goto LABEL_35;
          LODWORD(v18) = memcmp(v23, (const void *)v15, v14);
          v24 = v43;
          if ( !(_DWORD)v18 )
            goto LABEL_35;
        }
        LOBYTE(v18) = v23 + 1 == 0;
        if ( v20 == -1 || (LOBYTE(v18) = v23 + 2 == 0, v20 == -2) )
        {
          if ( !(_BYTE)v18 )
            goto LABEL_30;
        }
        else if ( v24 != na
               || (LOBYTE(v18) = na, na) && (LODWORD(v18) = memcmp(v23, (const void *)v20, na), (_DWORD)v18) )
        {
LABEL_30:
          sub_16F3520(a1, v21, v47, v19, v24);
          v25 = v47[0];
          v26 = *(_QWORD **)v21;
          *(_QWORD *)v21 = 0;
          v27 = *(_QWORD **)v25;
          *(_QWORD *)v25 = v26;
          if ( v27 )
          {
            if ( (_QWORD *)*v27 != v27 + 2 )
            {
              v34 = v25;
              v40 = v27;
              j_j___libc_free_0(*v27, v27[2] + 1LL);
              v25 = v34;
              v27 = v40;
            }
            v41 = v25;
            j_j___libc_free_0(v27, 32);
            v25 = v41;
          }
          *(__m128i *)(v25 + 8) = _mm_loadu_si128((const __m128i *)(v21 + 8));
          sub_16F2270(v25 + 24, (unsigned __int8 *)(v21 + 24));
          ++*(_DWORD *)(a1 + 16);
          LOBYTE(v18) = sub_16F2AA0((_QWORD *)(v21 + 24));
        }
LABEL_35:
        v28 = *(_QWORD **)v21;
        if ( *(_QWORD *)v21 )
        {
          if ( (_QWORD *)*v28 != v28 + 2 )
          {
            v42 = *(_QWORD **)v21;
            j_j___libc_free_0(*v28, v28[2] + 1LL);
            v28 = v42;
          }
          LOBYTE(v18) = j_j___libc_free_0(v28, 32);
        }
        v21 += 64;
        if ( v22 == v21 )
          goto LABEL_40;
      }
    }
  }
  else
  {
    sub_16F2420(v47, (unsigned __int8 *)0xFFFFFFFFFFFFFFFELL, 0);
    sub_16F25B0(&v44, (__int64)v47);
    v20 = v45;
    v30 = v44;
    na = v46;
    v18 = v48;
    if ( (_QWORD *)v47[0] != v48 )
      LOBYTE(v18) = j_j___libc_free_0(v47[0], v48[0] + 1LL);
    if ( v5 != a3 )
      goto LABEL_23;
LABEL_40:
    if ( v30 )
    {
      if ( *v30 != (unsigned __int8 *)(v30 + 2) )
        j_j___libc_free_0(*v30, v30[2] + 1);
      LOBYTE(v18) = j_j___libc_free_0(v30, 32);
    }
  }
  if ( v32 )
  {
    if ( *v32 != (unsigned __int8 *)(v32 + 2) )
      j_j___libc_free_0(*v32, v32[2] + 1);
    LOBYTE(v18) = j_j___libc_free_0(v32, 32);
  }
  return (char)v18;
}
