// Function: sub_C6B900
// Address: 0xc6b900
//
char __fastcall sub_C6B900(__int64 a1)
{
  _QWORD *v1; // rax
  size_t v2; // r14
  __int64 v3; // rbx
  _QWORD *v4; // r15
  _QWORD *v5; // r12
  __int64 v6; // r13
  __int64 v7; // r8
  _QWORD *v8; // r8
  char *v9; // rdi
  __int64 *v11; // [rsp-E0h] [rbp-E0h]
  __int64 *v12; // [rsp-D0h] [rbp-D0h]
  size_t v13; // [rsp-C8h] [rbp-C8h]
  __int64 v14; // [rsp-C0h] [rbp-C0h]
  _QWORD *v15; // [rsp-C0h] [rbp-C0h]
  __int64 *v16; // [rsp-78h] [rbp-78h] BYREF
  __int64 v17; // [rsp-70h] [rbp-70h]
  size_t v18; // [rsp-68h] [rbp-68h]
  __int64 v19[2]; // [rsp-58h] [rbp-58h] BYREF
  _QWORD v20[9]; // [rsp-48h] [rbp-48h] BYREF

  LODWORD(v1) = *(_DWORD *)(a1 + 24);
  if ( !(_DWORD)v1 )
    return (char)v1;
  v2 = 0;
  v3 = -1;
  v12 = 0;
  if ( !(unsigned __int8)sub_C6A630((char *)0xFFFFFFFFFFFFFFFFLL, 0, 0) )
  {
    sub_C6B0E0(v19, -1, 0);
    sub_C6B270(&v16, (__int64)v19);
    v3 = v17;
    v2 = v18;
    v12 = v16;
    if ( (_QWORD *)v19[0] != v20 )
      j_j___libc_free_0(v19[0], v20[0] + 1LL);
  }
  LOBYTE(v1) = sub_C6A630((char *)0xFFFFFFFFFFFFFFFELL, 0, 0);
  if ( (_BYTE)v1 )
  {
    v4 = *(_QWORD **)(a1 + 8);
    v5 = &v4[8 * (unsigned __int64)*(unsigned int *)(a1 + 24)];
    if ( v4 != v5 )
    {
      v13 = 0;
      v6 = -2;
      v11 = 0;
      goto LABEL_14;
    }
    goto LABEL_31;
  }
  sub_C6B0E0(v19, -2, 0);
  sub_C6B270(&v16, (__int64)v19);
  v6 = v17;
  v11 = v16;
  v13 = v18;
  v1 = v20;
  if ( (_QWORD *)v19[0] != v20 )
    LOBYTE(v1) = j_j___libc_free_0(v19[0], v20[0] + 1LL);
  v4 = *(_QWORD **)(a1 + 8);
  v5 = &v4[8 * (unsigned __int64)*(unsigned int *)(a1 + 24)];
  while ( v4 != v5 )
  {
LABEL_14:
    v9 = (char *)v4[1];
    v7 = v4[2];
    LOBYTE(v1) = v9 + 1 == 0;
    if ( v3 != -1 )
    {
      LOBYTE(v1) = v9 + 2 == 0;
      if ( v3 != -2 )
      {
        if ( v7 != v2 )
          goto LABEL_17;
        v14 = v4[2];
        if ( !v2 )
          goto LABEL_9;
        LODWORD(v1) = memcmp(v9, (const void *)v3, v2);
        v7 = v14;
        LOBYTE(v1) = (_DWORD)v1 == 0;
      }
    }
    if ( (_BYTE)v1 )
      goto LABEL_9;
LABEL_17:
    LOBYTE(v1) = v9 + 1 == 0;
    if ( v6 == -1 )
      goto LABEL_22;
    LOBYTE(v1) = v9 + 2 == 0;
    if ( v6 == -2 )
      goto LABEL_22;
    LOBYTE(v1) = v13;
    if ( v7 != v13 )
      goto LABEL_23;
    if ( v13 )
    {
      LOBYTE(v1) = memcmp(v9, (const void *)v6, v13) == 0;
LABEL_22:
      if ( !(_BYTE)v1 )
LABEL_23:
        LOBYTE(v1) = sub_C6BC50(v4 + 3);
    }
LABEL_9:
    v8 = (_QWORD *)*v4;
    if ( *v4 )
    {
      if ( (_QWORD *)*v8 != v8 + 2 )
      {
        v15 = (_QWORD *)*v4;
        j_j___libc_free_0(*v8, v8[2] + 1LL);
        v8 = v15;
      }
      LOBYTE(v1) = j_j___libc_free_0(v8, 32);
    }
    v4 += 8;
  }
  if ( v11 )
  {
    if ( (__int64 *)*v11 != v11 + 2 )
      j_j___libc_free_0(*v11, v11[2] + 1);
    LOBYTE(v1) = j_j___libc_free_0(v11, 32);
  }
LABEL_31:
  if ( v12 )
  {
    if ( (__int64 *)*v12 != v12 + 2 )
      j_j___libc_free_0(*v12, v12[2] + 1);
    LOBYTE(v1) = j_j___libc_free_0(v12, 32);
  }
  return (char)v1;
}
