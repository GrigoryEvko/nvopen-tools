// Function: sub_19B6370
// Address: 0x19b6370
//
void __fastcall sub_19B6370(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // r15
  char *v5; // rbx
  char *v6; // r12
  char *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // [rsp+8h] [rbp-238h] BYREF
  __m128i v11[2]; // [rsp+10h] [rbp-230h] BYREF
  _QWORD v12[11]; // [rsp+30h] [rbp-210h] BYREF
  char *v13; // [rsp+88h] [rbp-1B8h]
  unsigned int v14; // [rsp+90h] [rbp-1B0h]
  char v15; // [rsp+98h] [rbp-1A8h] BYREF

  v2 = sub_15E0530(*a1);
  if ( sub_1602790(v2)
    || (v8 = sub_15E0530(*a1),
        v9 = sub_16033E0(v8),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v9 + 48LL))(v9)) )
  {
    v3 = *a2;
    v4 = **(_QWORD **)(v3 + 32);
    sub_13FD840(&v10, v3);
    sub_15C9090((__int64)v11, &v10);
    sub_15CA330((__int64)v12, (__int64)"loop-unroll", (__int64)"ProfitableToRTUnroll", 20, v11, v4);
    if ( v10 )
      sub_161E7C0((__int64)&v10, v10);
    sub_15CAB20((__int64)v12, "      Failed : Not innermost loop (ProfitableToRTUnroll returns false)", 0x46u);
    sub_143AA50(a1, (__int64)v12);
    v5 = v13;
    v12[0] = &unk_49ECF68;
    v6 = &v13[88 * v14];
    if ( v13 != v6 )
    {
      do
      {
        v6 -= 88;
        v7 = (char *)*((_QWORD *)v6 + 4);
        if ( v7 != v6 + 48 )
          j_j___libc_free_0(v7, *((_QWORD *)v6 + 6) + 1LL);
        if ( *(char **)v6 != v6 + 16 )
          j_j___libc_free_0(*(_QWORD *)v6, *((_QWORD *)v6 + 2) + 1LL);
      }
      while ( v5 != v6 );
      v6 = v13;
    }
    if ( v6 != &v15 )
      _libc_free((unsigned __int64)v6);
  }
}
