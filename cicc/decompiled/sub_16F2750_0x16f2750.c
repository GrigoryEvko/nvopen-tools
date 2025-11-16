// Function: sub_16F2750
// Address: 0x16f2750
//
char __fastcall sub_16F2750(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  size_t v5; // r15
  __int64 v6; // r14
  __int64 v7; // rcx
  int v8; // r8d
  _QWORD *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r9
  size_t v12; // r11
  char *v13; // r13
  __int64 v14; // r8
  unsigned __int8 **v16; // [rsp+8h] [rbp-D8h]
  size_t v17; // [rsp+10h] [rbp-D0h]
  __int64 v18; // [rsp+18h] [rbp-C8h]
  size_t v19; // [rsp+18h] [rbp-C8h]
  size_t v20; // [rsp+18h] [rbp-C8h]
  unsigned __int8 **v21; // [rsp+20h] [rbp-C0h]
  __int64 v22; // [rsp+28h] [rbp-B8h]
  __int64 v23; // [rsp+28h] [rbp-B8h]
  __int64 v24; // [rsp+28h] [rbp-B8h]
  unsigned __int8 **v25; // [rsp+70h] [rbp-70h] BYREF
  __int64 v26; // [rsp+78h] [rbp-68h]
  size_t v27; // [rsp+80h] [rbp-60h]
  __int64 v28[2]; // [rsp+90h] [rbp-50h] BYREF
  _QWORD v29[8]; // [rsp+A0h] [rbp-40h] BYREF

  v5 = 0;
  v6 = -1;
  v21 = 0;
  if ( !(unsigned __int8)sub_16F23B0((unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0, 0, a4, a5) )
  {
    sub_16F2420(v28, (unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0);
    sub_16F25B0(&v25, (__int64)v28);
    v6 = v26;
    v5 = v27;
    v21 = v25;
    if ( (_QWORD *)v28[0] != v29 )
      j_j___libc_free_0(v28[0], v29[0] + 1LL);
  }
  LOBYTE(v9) = sub_16F23B0((unsigned __int8 *)0xFFFFFFFFFFFFFFFELL, 0, 0, v7, v8);
  if ( !(_BYTE)v9 )
  {
    sub_16F2420(v28, (unsigned __int8 *)0xFFFFFFFFFFFFFFFELL, 0);
    sub_16F25B0(&v25, (__int64)v28);
    v11 = v26;
    v12 = v27;
    v16 = v25;
    v9 = v29;
    if ( (_QWORD *)v28[0] != v29 )
    {
      v20 = v27;
      v24 = v26;
      LOBYTE(v9) = j_j___libc_free_0(v28[0], v29[0] + 1LL);
      v12 = v20;
      v11 = v24;
    }
    v10 = *(_QWORD *)(a1 + 16);
    if ( *(_QWORD *)(a1 + 24) == v10 )
    {
LABEL_10:
      if ( v16 )
      {
        if ( *v16 != (unsigned __int8 *)(v16 + 2) )
          j_j___libc_free_0(*v16, v16[2] + 1);
        LOBYTE(v9) = j_j___libc_free_0(v16, 32);
      }
      goto LABEL_14;
    }
    while ( 1 )
    {
LABEL_4:
      v13 = *(char **)(v10 + 8);
      v14 = *(_QWORD *)(v10 + 16);
      LOBYTE(v9) = v13 + 1 == 0;
      if ( v6 == -1 || (LOBYTE(v9) = v13 + 2 == 0, v6 == -2) )
      {
        if ( (_BYTE)v9 )
          goto LABEL_19;
      }
      else if ( v5 == v14 )
      {
        v22 = *(_QWORD *)(v10 + 16);
        if ( !v5 )
          goto LABEL_19;
        v17 = v12;
        v18 = v11;
        LODWORD(v9) = memcmp(v13, (const void *)v6, v5);
        v11 = v18;
        v12 = v17;
        v14 = v22;
        if ( !(_DWORD)v9 )
          goto LABEL_19;
      }
      LOBYTE(v9) = v13 + 1 == 0;
      if ( v11 == -1 || (LOBYTE(v9) = v13 + 2 == 0, v11 == -2) )
      {
        if ( !(_BYTE)v9 )
          goto LABEL_10;
      }
      else
      {
        if ( v14 != v12 )
          goto LABEL_10;
        if ( v12 )
        {
          v19 = v12;
          v23 = v11;
          LODWORD(v9) = memcmp(v13, (const void *)v11, v12);
          v11 = v23;
          v12 = v19;
          if ( (_DWORD)v9 )
            goto LABEL_10;
        }
      }
LABEL_19:
      v10 += 64;
      *(_QWORD *)(a1 + 16) = v10;
      if ( *(_QWORD *)(a1 + 24) == v10 )
        goto LABEL_10;
    }
  }
  v10 = *(_QWORD *)(a1 + 16);
  v11 = -2;
  v12 = 0;
  v16 = 0;
  if ( *(_QWORD *)(a1 + 24) != v10 )
    goto LABEL_4;
LABEL_14:
  if ( v21 )
  {
    if ( *v21 != (unsigned __int8 *)(v21 + 2) )
      j_j___libc_free_0(*v21, v21[2] + 1);
    LOBYTE(v9) = j_j___libc_free_0(v21, 32);
  }
  return (char)v9;
}
