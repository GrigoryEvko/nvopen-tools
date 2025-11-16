// Function: sub_23B08D0
// Address: 0x23b08d0
//
void __fastcall sub_23B08D0(__int64 a1, __int64 a2, __int64 a3)
{
  char *v3; // r13
  __int64 v4; // r12
  char *v5; // r14
  const char *v6; // r15
  char *v7; // rsi
  __int64 v8; // rdx
  char *v9; // r15
  size_t v10; // rax
  __int64 v11; // rbx
  __int64 v12; // [rsp+18h] [rbp-B8h]
  const char *v13[2]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD v14[2]; // [rsp+30h] [rbp-A0h] BYREF
  const char *v15[2]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v16[2]; // [rsp+50h] [rbp-80h] BYREF
  const char *v17[2]; // [rsp+60h] [rbp-70h] BYREF
  _QWORD v18[2]; // [rsp+70h] [rbp-60h] BYREF
  __m128i v19; // [rsp+80h] [rbp-50h] BYREF
  __int64 v20; // [rsp+90h] [rbp-40h] BYREF

  v3 = "\n";
  v4 = 1;
  if ( a2 )
  {
    v3 = *(char **)(a2 + 32);
    v4 = *(_QWORD *)(a2 + 40);
  }
  v12 = 1;
  v5 = "\n";
  if ( a3 )
  {
    v5 = *(char **)(a3 + 32);
    v12 = *(_QWORD *)(a3 + 40);
  }
  v6 = "\x1B[31m-%l\x1B[0m\n";
  if ( !*(_BYTE *)(*(_QWORD *)a1 + 48LL) )
    v6 = "-%l\n";
  v13[0] = (const char *)v14;
  v7 = (char *)v6;
  v8 = (__int64)&v6[strlen(v6)];
  v9 = "\x1B[32m+%l\x1B[0m\n";
  sub_23AE760((__int64 *)v13, v7, v8);
  if ( !*(_BYTE *)(*(_QWORD *)a1 + 48LL) )
    v9 = "+%l\n";
  v15[0] = (const char *)v16;
  v10 = strlen(v9);
  sub_23AE760((__int64 *)v15, v9, (__int64)&v9[v10]);
  v17[0] = (const char *)v18;
  sub_23AE760((__int64 *)v17, " %l\n", (__int64)"");
  v11 = *(_QWORD *)(*(_QWORD *)a1 + 40LL);
  sub_BC7A80(
    &v19,
    (__int64)v3,
    v4,
    (__int64)v5,
    v12,
    (__int64)&v19,
    v13[0],
    (__int64)v13[1],
    v15[0],
    (__int64)v15[1],
    v17[0],
    (__int64)v17[1]);
  sub_CB6200(v11, (unsigned __int8 *)v19.m128i_i64[0], v19.m128i_u64[1]);
  if ( (__int64 *)v19.m128i_i64[0] != &v20 )
    j_j___libc_free_0(v19.m128i_u64[0]);
  if ( (_QWORD *)v17[0] != v18 )
    j_j___libc_free_0((unsigned __int64)v17[0]);
  if ( (_QWORD *)v15[0] != v16 )
    j_j___libc_free_0((unsigned __int64)v15[0]);
  if ( (_QWORD *)v13[0] != v14 )
    j_j___libc_free_0((unsigned __int64)v13[0]);
}
