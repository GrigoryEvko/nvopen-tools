// Function: sub_29F4100
// Address: 0x29f4100
//
void __fastcall sub_29F4100(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char **v6; // r13
  char *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // r8
  bool v11; // cc
  __int64 v12; // rcx
  char *v13; // rdx
  __int64 v14; // rdi
  char *v15; // [rsp+0h] [rbp-80h] BYREF
  __int64 v16; // [rsp+8h] [rbp-78h]
  __int64 v17; // [rsp+10h] [rbp-70h]
  _BYTE v18[104]; // [rsp+18h] [rbp-68h] BYREF

  v6 = (char **)a1;
  v7 = v18;
  v8 = *(_QWORD *)(a1 + 8);
  v15 = v18;
  v16 = 0;
  v17 = 64;
  if ( v8 )
  {
    sub_29F3DD0((__int64)&v15, (char **)a1, v8, a4, a5, a6);
    v8 = v16;
    v7 = v15;
  }
  while ( 1 )
  {
    v9 = (__int64)*(v6 - 10);
    v10 = (__int64)&v7[v8];
    v11 = v8 <= v9;
    v12 = (__int64)&v7[v9];
    v13 = *(v6 - 11);
    if ( v11 )
      v12 = v10;
    if ( (char *)v12 == v7 )
      break;
    while ( 1 )
    {
      a6 = (unsigned __int8)*v13;
      if ( *v7 < (char)a6 )
        break;
      if ( *v7 > (char)a6 )
        goto LABEL_12;
      ++v7;
      ++v13;
      if ( (char *)v12 == v7 )
        goto LABEL_11;
    }
LABEL_10:
    v14 = (__int64)v6;
    v6 -= 11;
    sub_29F3DD0(v14, v6, (__int64)v13, v12, v10, a6);
    v7 = v15;
    v8 = v16;
  }
LABEL_11:
  if ( v13 != &(*(v6 - 11))[v9] )
    goto LABEL_10;
LABEL_12:
  sub_29F3DD0((__int64)v6, &v15, (__int64)v13, v12, v10, a6);
  if ( v15 != v18 )
    _libc_free((unsigned __int64)v15);
}
