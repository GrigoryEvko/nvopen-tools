// Function: sub_18695B0
// Address: 0x18695b0
//
__int64 __fastcall sub_18695B0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r14
  unsigned __int64 v3; // rax
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 *v7; // rsi
  __int64 v8; // rbx
  __int64 *v9; // rsi
  char v10; // al
  char *v11; // rbx
  char *v12; // r15
  char *v13; // rdi
  __int64 **v14; // rax
  __int64 v15; // r9
  __int64 v16; // r12
  __int64 *v17; // r13
  char v19; // [rsp+Fh] [rbp-271h]
  int v20; // [rsp+20h] [rbp-260h]
  int v21; // [rsp+28h] [rbp-258h]
  __int64 *v22[2]; // [rsp+30h] [rbp-250h] BYREF
  __int64 *v23; // [rsp+40h] [rbp-240h]
  __m128i v24; // [rsp+50h] [rbp-230h] BYREF
  char v25; // [rsp+60h] [rbp-220h]
  _QWORD v26[2]; // [rsp+70h] [rbp-210h] BYREF
  __int64 (__fastcall *v27)(_QWORD *, _QWORD *, int); // [rsp+80h] [rbp-200h]
  __int64 (__fastcall *v28)(__int64, __int64); // [rsp+88h] [rbp-1F8h]
  char *v29; // [rsp+C8h] [rbp-1B8h]
  unsigned int v30; // [rsp+D0h] [rbp-1B0h]
  char v31; // [rsp+D8h] [rbp-1A8h] BYREF

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = (a2 & 0xFFFFFFFFFFFFFFF8LL) - 72;
  v20 = a2;
  v5 = *(_QWORD *)(a1 + 344);
  if ( (a2 & 4) != 0 )
    v3 = (a2 & 0xFFFFFFFFFFFFFFF8LL) - 24;
  v6 = *(_QWORD *)v3;
  if ( *(_BYTE *)(*(_QWORD *)v3 + 16LL) )
    v6 = 0;
  v21 = sub_14A4050(v5, v6);
  v7 = *(__int64 **)(*(_QWORD *)(v2 + 40) + 56LL);
  if ( v7 + 9 == (__int64 *)(v7[9] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    sub_143A950(v22, v7);
    v26[0] = a1;
    v28 = sub_18694D0;
    v27 = sub_1869580;
    v14 = 0;
  }
  else
  {
    v8 = v7[10];
    v22[0] = 0;
    if ( v8 )
      v8 -= 24;
    sub_15C9090((__int64)&v24, v22);
    sub_15CA330((__int64)v26, (__int64)"inline", (__int64)byte_3F871B3, 0, &v24, v8);
    v9 = v22[0];
    if ( v22[0] )
      sub_161E7C0((__int64)v22, (__int64)v22[0]);
    v10 = sub_15C8000((__int64)v26, (__int64)v9);
    v11 = v29;
    v19 = v10;
    v26[0] = &unk_49ECF68;
    v12 = &v29[88 * v30];
    if ( v29 != v12 )
    {
      do
      {
        v12 -= 88;
        v13 = (char *)*((_QWORD *)v12 + 4);
        if ( v13 != v12 + 48 )
          j_j___libc_free_0(v13, *((_QWORD *)v12 + 6) + 1LL);
        if ( *(char **)v12 != v12 + 16 )
          j_j___libc_free_0(*(_QWORD *)v12, *((_QWORD *)v12 + 2) + 1LL);
      }
      while ( v11 != v12 );
      v12 = v29;
    }
    if ( v12 != &v31 )
      _libc_free((unsigned __int64)v12);
    sub_143A950(v22, *(__int64 **)(*(_QWORD *)(v2 + 40) + 56LL));
    v26[0] = a1;
    v28 = sub_18694D0;
    v27 = sub_1869580;
    v14 = 0;
    if ( v19 )
      v14 = v22;
  }
  v15 = *(_QWORD *)(a1 + 168);
  v25 = 0;
  v16 = sub_385A210(v20, (int)a1 + 276, v21, (unsigned int)v26, (unsigned int)&v24, v15, (__int64)v14);
  if ( v27 )
    v27(v26, v26, 3);
  v17 = v23;
  if ( v23 )
  {
    sub_1368A00(v23);
    j_j___libc_free_0(v17, 8);
  }
  return v16;
}
