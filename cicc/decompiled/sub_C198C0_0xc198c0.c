// Function: sub_C198C0
// Address: 0xc198c0
//
void __fastcall sub_C198C0(__int64 a1, __int64 a2, __int64 a3, char *a4)
{
  __int64 v4; // rax
  unsigned __int64 v6; // r15
  unsigned __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r9
  char *v12; // rax
  __int64 v13; // r13
  __int64 v14; // r14
  __int64 v15; // rbx
  char *v16; // rdx
  int v17; // ecx
  __int64 v18; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v19; // [rsp+18h] [rbp-E8h]
  char *v20; // [rsp+30h] [rbp-D0h]
  char *v21; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v22; // [rsp+40h] [rbp-C0h]
  _BYTE v23[56]; // [rsp+48h] [rbp-B8h] BYREF
  char *v24; // [rsp+80h] [rbp-80h] BYREF
  __int64 v25; // [rsp+88h] [rbp-78h] BYREF
  __int64 v26; // [rsp+90h] [rbp-70h] BYREF
  _BYTE v27[104]; // [rsp+98h] [rbp-68h] BYREF

  v4 = a2 - a1;
  v19 = a2;
  v18 = a3;
  if ( a2 - a1 <= 1152 )
    return;
  if ( !a3 )
  {
    v8 = a2;
    goto LABEL_16;
  }
  while ( 2 )
  {
    v6 = v19;
    --v18;
    v7 = a1 + 72;
    sub_C18F90(
      (__int64 *)a1,
      a1 + 72,
      (_QWORD *)(a1 + 72 * ((__int64)(0x8E38E38E38E38E39LL * (v4 >> 3)) >> 1)),
      (_QWORD *)(v19 - 72),
      (__int64)a4);
    while ( 1 )
    {
      v8 = v7;
      if ( sub_C185F0((__int64)a4, v7, a1) )
        goto LABEL_4;
      do
        v6 -= 72LL;
      while ( sub_C185F0((__int64)a4, a1, v6) );
      if ( v7 >= v6 )
        break;
      v9 = *(_QWORD *)v7;
      v10 = *(_QWORD *)v6;
      v11 = v7 + 8;
      v24 = (char *)&v26;
      *(_QWORD *)v7 = v10;
      *(_QWORD *)v6 = v9;
      v25 = 0xC00000000LL;
      if ( *(_DWORD *)(v7 + 16) )
      {
        sub_C15E20((__int64)&v24, (char **)(v7 + 8));
        v11 = v7 + 8;
      }
      sub_C15E20(v11, (char **)(v6 + 8));
      sub_C15E20(v6 + 8, &v24);
      if ( v24 != (char *)&v26 )
        _libc_free(v24, &v24);
LABEL_4:
      v7 += 72LL;
    }
    sub_C198C0(v7, v19, v18, a4);
    v4 = v7 - a1;
    if ( (__int64)(v7 - a1) > 1152 )
    {
      if ( v18 )
      {
        v19 = v7;
        continue;
      }
LABEL_16:
      v24 = a4;
      sub_C19720(a1, v8, (__int64 *)&v24);
      v12 = a4;
      v13 = v8 - 64;
      v14 = (__int64)v12;
      do
      {
        v16 = *(char **)(v13 - 8);
        v17 = *(_DWORD *)(v13 + 8);
        v21 = v23;
        v20 = v16;
        v22 = 0xC00000000LL;
        if ( v17 )
          sub_C15E20((__int64)&v21, (char **)v13);
        *(_QWORD *)(v13 - 8) = *(_QWORD *)a1;
        sub_C15E20(v13, (char **)(a1 + 8));
        v24 = v20;
        v25 = (__int64)v27;
        v26 = 0xC00000000LL;
        if ( (_DWORD)v22 )
          sub_C15E20((__int64)&v25, &v21);
        v15 = v13 - 8 - a1;
        sub_C19490(a1, 0, 0x8E38E38E38E38E39LL * (v15 >> 3), (__int64 *)&v24, v14);
        if ( (_BYTE *)v25 != v27 )
          _libc_free(v25, 0);
        if ( v21 != v23 )
          _libc_free(v21, 0);
        v13 -= 72;
      }
      while ( v15 > 72 );
    }
    break;
  }
}
