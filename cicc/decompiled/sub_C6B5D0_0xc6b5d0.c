// Function: sub_C6B5D0
// Address: 0xc6b5d0
//
char __fastcall sub_C6B5D0(__int64 a1)
{
  size_t v1; // r15
  __int64 v2; // r14
  _QWORD *v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r9
  size_t v6; // r11
  __int64 v7; // r8
  char *v8; // r13
  __int64 *v10; // [rsp+8h] [rbp-D8h]
  __int64 *v11; // [rsp+10h] [rbp-D0h]
  size_t n; // [rsp+18h] [rbp-C8h]
  void *s2; // [rsp+20h] [rbp-C0h]
  void *s2a; // [rsp+20h] [rbp-C0h]
  void *s2b; // [rsp+20h] [rbp-C0h]
  __int64 v16; // [rsp+28h] [rbp-B8h]
  __int64 v17; // [rsp+28h] [rbp-B8h]
  __int64 v18; // [rsp+28h] [rbp-B8h]
  __int64 *v19; // [rsp+70h] [rbp-70h] BYREF
  __int64 v20; // [rsp+78h] [rbp-68h]
  void *v21; // [rsp+80h] [rbp-60h]
  __int64 v22[2]; // [rsp+90h] [rbp-50h] BYREF
  _QWORD v23[8]; // [rsp+A0h] [rbp-40h] BYREF

  v1 = 0;
  v2 = -1;
  v11 = 0;
  if ( !(unsigned __int8)sub_C6A630((char *)0xFFFFFFFFFFFFFFFFLL, 0, 0) )
  {
    sub_C6B0E0(v22, -1, 0);
    sub_C6B270(&v19, (__int64)v22);
    v2 = v20;
    v1 = (size_t)v21;
    v11 = v19;
    if ( (_QWORD *)v22[0] != v23 )
      j_j___libc_free_0(v22[0], v23[0] + 1LL);
  }
  LOBYTE(v3) = sub_C6A630((char *)0xFFFFFFFFFFFFFFFELL, 0, 0);
  if ( !(_BYTE)v3 )
  {
    sub_C6B0E0(v22, -2, 0);
    sub_C6B270(&v19, (__int64)v22);
    v5 = v20;
    v6 = (size_t)v21;
    v10 = v19;
    v3 = v23;
    if ( (_QWORD *)v22[0] != v23 )
    {
      s2b = v21;
      v18 = v20;
      LOBYTE(v3) = j_j___libc_free_0(v22[0], v23[0] + 1LL);
      v6 = (size_t)s2b;
      v5 = v18;
    }
    v4 = *(_QWORD *)(a1 + 16);
    if ( *(_QWORD *)(a1 + 24) == v4 )
    {
LABEL_15:
      if ( v10 )
      {
        if ( (__int64 *)*v10 != v10 + 2 )
          j_j___libc_free_0(*v10, v10[2] + 1);
        LOBYTE(v3) = j_j___libc_free_0(v10, 32);
      }
      goto LABEL_19;
    }
    while ( 1 )
    {
LABEL_9:
      v8 = *(char **)(v4 + 8);
      v7 = *(_QWORD *)(v4 + 16);
      LOBYTE(v3) = v8 + 1 == 0;
      if ( v2 != -1 )
      {
        LOBYTE(v3) = v8 + 2 == 0;
        if ( v2 != -2 )
        {
          if ( v1 != v7 )
            goto LABEL_12;
          v16 = *(_QWORD *)(v4 + 16);
          if ( !v1 )
            goto LABEL_8;
          n = v6;
          s2 = (void *)v5;
          LODWORD(v3) = memcmp(v8, (const void *)v2, v1);
          v7 = v16;
          v6 = n;
          v5 = (__int64)s2;
          LOBYTE(v3) = (_DWORD)v3 == 0;
        }
      }
      if ( (_BYTE)v3 )
        goto LABEL_8;
LABEL_12:
      LOBYTE(v3) = v8 + 1 == 0;
      if ( v5 != -1 )
      {
        LOBYTE(v3) = v8 + 2 == 0;
        if ( v5 != -2 )
        {
          if ( v7 != v6 )
            goto LABEL_15;
          if ( !v6 )
            goto LABEL_8;
          s2a = (void *)v6;
          v17 = v5;
          LODWORD(v3) = memcmp(v8, (const void *)v5, v6);
          v6 = (size_t)s2a;
          v5 = v17;
          LOBYTE(v3) = (_DWORD)v3 == 0;
        }
      }
      if ( !(_BYTE)v3 )
        goto LABEL_15;
LABEL_8:
      v4 += 64;
      *(_QWORD *)(a1 + 16) = v4;
      if ( *(_QWORD *)(a1 + 24) == v4 )
        goto LABEL_15;
    }
  }
  v4 = *(_QWORD *)(a1 + 16);
  v5 = -2;
  v6 = 0;
  v10 = 0;
  if ( *(_QWORD *)(a1 + 24) != v4 )
    goto LABEL_9;
LABEL_19:
  if ( v11 )
  {
    if ( (__int64 *)*v11 != v11 + 2 )
      j_j___libc_free_0(*v11, v11[2] + 1);
    LOBYTE(v3) = j_j___libc_free_0(v11, 32);
  }
  return (char)v3;
}
