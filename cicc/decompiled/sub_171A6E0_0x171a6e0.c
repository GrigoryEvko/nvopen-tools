// Function: sub_171A6E0
// Address: 0x171a6e0
//
__int64 __fastcall sub_171A6E0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r15
  __int64 *v3; // r14
  void *v6; // rax
  void *v7; // r13
  __int64 *v8; // rsi
  _QWORD *v9; // rdi
  void *v11; // rax
  void *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // [rsp+0h] [rbp-40h]
  __int64 v17; // [rsp+8h] [rbp-38h]

  v2 = (__int64 *)(a2 + 8);
  v3 = (__int64 *)(a1 + 16);
  v6 = sub_16982C0();
  v7 = v6;
  if ( !*(_BYTE *)a1 )
  {
    v8 = (__int64 *)(a2 + 8);
    v9 = (_QWORD *)(a1 + 16);
    if ( *(void **)(a2 + 8) != v6 )
    {
LABEL_3:
      sub_16986C0(v9, v8);
      goto LABEL_4;
    }
LABEL_13:
    sub_169C6E0(v9, (__int64)v8);
    goto LABEL_4;
  }
  v11 = *(void **)(a2 + 8);
  if ( *(void **)(a1 + 16) != v7 )
  {
    if ( v11 != v7 )
    {
      sub_1698680((__int64 *)(a1 + 16), (__int64 *)(a2 + 8));
      goto LABEL_4;
    }
    if ( v3 == v2 )
      goto LABEL_4;
    sub_1698460(a1 + 16);
    v12 = *(void **)(a2 + 8);
    goto LABEL_10;
  }
  if ( v11 == v7 )
  {
    sub_16A0170((__int64 *)(a1 + 16), (__int64 *)(a2 + 8));
    goto LABEL_4;
  }
  if ( v3 != v2 )
  {
    v13 = *(_QWORD *)(a1 + 24);
    if ( !v13 )
      goto LABEL_11;
    v14 = 32LL * *(_QWORD *)(v13 - 8);
    v15 = v13 + v14;
    if ( v13 != v13 + v14 )
    {
      do
      {
        v16 = v13;
        v17 = v15 - 32;
        sub_127D120((_QWORD *)(v15 - 24));
        v15 = v17;
        v13 = v16;
      }
      while ( v16 != v17 );
    }
    j_j_j___libc_free_0_0(v13 - 8);
    v12 = *(void **)(a2 + 8);
LABEL_10:
    if ( v12 == v7 )
    {
      v8 = v2;
      v9 = (_QWORD *)(a1 + 16);
      goto LABEL_13;
    }
LABEL_11:
    v8 = v2;
    v9 = (_QWORD *)(a1 + 16);
    goto LABEL_3;
  }
LABEL_4:
  *(_WORD *)a1 = 257;
  return 257;
}
