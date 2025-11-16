// Function: sub_18A4B90
// Address: 0x18a4b90
//
void __fastcall sub_18A4B90(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int64 *a8,
        unsigned int *a9,
        unsigned int *a10)
{
  __int64 v10; // rax
  char *v11; // rbx
  char *v12; // r12
  char *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  _QWORD v16[2]; // [rsp+0h] [rbp-270h] BYREF
  _QWORD v17[2]; // [rsp+10h] [rbp-260h] BYREF
  _QWORD *v18; // [rsp+20h] [rbp-250h] BYREF
  _QWORD v19[6]; // [rsp+30h] [rbp-240h] BYREF
  _QWORD v20[11]; // [rsp+60h] [rbp-210h] BYREF
  char *v21; // [rsp+B8h] [rbp-1B8h]
  unsigned int v22; // [rsp+C0h] [rbp-1B0h]
  char v23; // [rsp+C8h] [rbp-1A8h] BYREF

  v10 = sub_15E0530(*a1);
  if ( sub_1602790(v10)
    || (v14 = sub_15E0530(*a1),
        v15 = sub_16033E0(v14),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v15 + 48LL))(v15)) )
  {
    sub_15CA700((__int64)v20, (__int64)"sample-profile", (__int64)"AppliedSamples", 14, a7);
    sub_15CAB20((__int64)v20, "Applied ", 8u);
    sub_15C9D40((__int64)v16, "NumSamples", 10, *a8);
    sub_18A4190((__int64)v20, (__int64)v16);
    if ( v18 != v19 )
      j_j___libc_free_0(v18, v19[0] + 1LL);
    if ( (_QWORD *)v16[0] != v17 )
      j_j___libc_free_0(v16[0], v17[0] + 1LL);
    sub_15CAB20((__int64)v20, " samples from profile (offset: ", 0x1Fu);
    sub_15C9C50((__int64)v16, "LineOffset", 10, *a9);
    sub_18A4190((__int64)v20, (__int64)v16);
    if ( v18 != v19 )
      j_j___libc_free_0(v18, v19[0] + 1LL);
    if ( (_QWORD *)v16[0] != v17 )
      j_j___libc_free_0(v16[0], v17[0] + 1LL);
    if ( *a10 )
    {
      sub_15CAB20((__int64)v20, ".", 1u);
      sub_15C9C50((__int64)v16, "Discriminator", 13, *a10);
      sub_18A4190((__int64)v20, (__int64)v16);
      sub_2240A30(&v18);
      sub_2240A30(v16);
    }
    sub_15CAB20((__int64)v20, ")", 1u);
    sub_143AA50(a1, (__int64)v20);
    v11 = v21;
    v20[0] = &unk_49ECF68;
    v12 = &v21[88 * v22];
    if ( v21 != v12 )
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
      v12 = v21;
    }
    if ( v12 != &v23 )
      _libc_free((unsigned __int64)v12);
  }
}
