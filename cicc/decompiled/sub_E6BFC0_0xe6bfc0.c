// Function: sub_E6BFC0
// Address: 0xe6bfc0
//
__int64 __fastcall sub_E6BFC0(_DWORD *a1, __int64 a2, char a3, unsigned __int8 a4)
{
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 v9; // rbx
  size_t v10; // rax
  char *v11; // rax
  char *i; // rdx
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rsi
  __int64 v15; // r12
  _QWORD v18[8]; // [rsp+10h] [rbp-110h] BYREF
  char *v19; // [rsp+50h] [rbp-D0h] BYREF
  size_t v20; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v21; // [rsp+60h] [rbp-C0h]
  _BYTE v22[184]; // [rsp+68h] [rbp-B8h] BYREF

  v19 = v22;
  v20 = 0;
  v21 = 128;
  sub_CA0EC0(a2, (__int64)&v19);
  v5 = v20;
  v6 = sub_E6B3F0((__int64)a1, v19, v20);
  v9 = v6;
  if ( !a3 )
    goto LABEL_12;
  do
  {
    v10 = v20;
    if ( v5 != v20 )
    {
      if ( v5 >= v20 )
      {
        if ( v5 > v21 )
        {
          sub_C8D290((__int64)&v19, v22, v5, 1u, v7, v8);
          v10 = v20;
        }
        v11 = &v19[v10];
        for ( i = &v19[v5]; i != v11; ++v11 )
        {
          if ( v11 )
            *v11 = 0;
        }
      }
      v20 = v5;
    }
    v18[5] = 0x100000000LL;
    v18[1] = 2;
    memset(&v18[2], 0, 24);
    v18[0] = &unk_49DD288;
    v18[6] = &v19;
    sub_CB5980((__int64)v18, 0, 0, 0);
    v13 = *(unsigned int *)(v9 + 16);
    *(_DWORD *)(v9 + 16) = v13 + 1;
    sub_CB59D0((__int64)v18, v13);
    v18[0] = &unk_49DD388;
    sub_CB5840((__int64)v18);
    v6 = sub_E6B3F0((__int64)a1, v19, v20);
LABEL_12:
    ;
  }
  while ( *(_BYTE *)(v6 + 20) );
  *(_BYTE *)(v6 + 20) = 1;
  v14 = v6;
  v15 = sub_E6BCB0(a1, v6, a4);
  if ( v19 != v22 )
    _libc_free(v19, v14);
  return v15;
}
