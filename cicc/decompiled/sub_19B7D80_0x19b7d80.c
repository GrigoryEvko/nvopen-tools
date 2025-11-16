// Function: sub_19B7D80
// Address: 0x19b7d80
//
void __fastcall sub_19B7D80(__int64 *a1, __int64 *a2, unsigned int *a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rax
  char *v7; // rbx
  char *v8; // r12
  char *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-288h]
  __int64 v13; // [rsp+18h] [rbp-278h] BYREF
  __m128i v14; // [rsp+20h] [rbp-270h] BYREF
  __int64 v15; // [rsp+30h] [rbp-260h] BYREF
  __int64 *v16; // [rsp+40h] [rbp-250h]
  __int64 v17; // [rsp+50h] [rbp-240h] BYREF
  _QWORD v18[11]; // [rsp+80h] [rbp-210h] BYREF
  char *v19; // [rsp+D8h] [rbp-1B8h]
  unsigned int v20; // [rsp+E0h] [rbp-1B0h]
  char v21; // [rsp+E8h] [rbp-1A8h] BYREF

  v4 = sub_15E0530(*a1);
  if ( sub_1602790(v4)
    || (v10 = sub_15E0530(*a1),
        v11 = sub_16033E0(v10),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v11 + 48LL))(v11)) )
  {
    v5 = *a2;
    v12 = **(_QWORD **)(v5 + 32);
    sub_13FD840(&v13, v5);
    sub_15C9090((__int64)&v14, &v13);
    sub_15CA330((__int64)v18, (__int64)"loop-unroll", (__int64)"ProfitableToRTUnroll", 20, &v14, v12);
    if ( v13 )
      sub_161E7C0((__int64)&v13, v13);
    sub_15CAB20((__int64)v18, "      Failed : loop body size ", 0x1Eu);
    sub_15C9C50((__int64)&v14, "LoopSize", 8, *a3);
    v6 = sub_17C2270((__int64)v18, (__int64)&v14);
    sub_15CAB20(v6, " is too large ", 0xEu);
    if ( v16 != &v17 )
      j_j___libc_free_0(v16, v17 + 1);
    if ( (__int64 *)v14.m128i_i64[0] != &v15 )
      j_j___libc_free_0(v14.m128i_i64[0], v15 + 1);
    sub_143AA50(a1, (__int64)v18);
    v7 = v19;
    v18[0] = &unk_49ECF68;
    v8 = &v19[88 * v20];
    if ( v19 != v8 )
    {
      do
      {
        v8 -= 88;
        v9 = (char *)*((_QWORD *)v8 + 4);
        if ( v9 != v8 + 48 )
          j_j___libc_free_0(v9, *((_QWORD *)v8 + 6) + 1LL);
        if ( *(char **)v8 != v8 + 16 )
          j_j___libc_free_0(*(_QWORD *)v8, *((_QWORD *)v8 + 2) + 1LL);
      }
      while ( v7 != v8 );
      v8 = v19;
    }
    if ( v8 != &v21 )
      _libc_free((unsigned __int64)v8);
  }
}
