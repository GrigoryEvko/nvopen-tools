// Function: sub_2412E00
// Address: 0x2412e00
//
__int64 __fastcall sub_2412E00(__int64 a1, unsigned __int64 a2, __int64 a3, __int16 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v8; // r14
  __int64 v9; // r9
  __int64 **v10; // r13
  __int64 (__fastcall *v11)(__int64, unsigned int, unsigned __int8 *, __int64); // rax
  __int64 v12; // r12
  int v14; // r13d
  _QWORD *v15; // rbx
  _QWORD *v16; // r13
  __int64 v17; // rdx
  unsigned int v18; // esi
  _WORD v19[24]; // [rsp-1B8h] [rbp-1B8h] BYREF
  _WORD v20[24]; // [rsp-188h] [rbp-188h] BYREF
  __int64 v21[2]; // [rsp-158h] [rbp-158h] BYREF
  _QWORD v22[16]; // [rsp-148h] [rbp-148h] BYREF
  _QWORD *v23; // [rsp-C8h] [rbp-C8h] BYREF
  unsigned int v24; // [rsp-C0h] [rbp-C0h]
  _QWORD v25[8]; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v26; // [rsp-78h] [rbp-78h]
  __int64 v27; // [rsp-70h] [rbp-70h]
  __int64 v28; // [rsp-68h] [rbp-68h]
  int v29; // [rsp-60h] [rbp-60h]
  void *v30; // [rsp-48h] [rbp-48h]

  if ( !a3 )
    BUG();
  sub_2412230((__int64)v21, *(_QWORD *)(a3 + 16), a3, a4, 0, a6, 0, 0);
  v8 = sub_240FA00(a1, a2, v21);
  sub_2412230((__int64)&v23, *(_QWORD *)(a3 + 16), a3, a4, 0, v9, 0, 0);
  v10 = *(__int64 ***)(a1 + 56);
  v19[16] = 257;
  if ( v10 == *((__int64 ***)v8 + 1) )
  {
    v12 = (__int64)v8;
    goto LABEL_8;
  }
  v11 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, __int64))(*(_QWORD *)v26 + 120LL);
  if ( v11 != sub_920130 )
  {
    v12 = v11(v26, 48u, v8, (__int64)v10);
    goto LABEL_7;
  }
  if ( *v8 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x30u) )
      v12 = sub_ADAB70(48, (unsigned __int64)v8, v10, 0);
    else
      v12 = sub_AA93C0(0x30u, (unsigned __int64)v8, (__int64)v10);
LABEL_7:
    if ( v12 )
      goto LABEL_8;
  }
  v20[16] = 257;
  v12 = sub_B51D30(48, (__int64)v8, (__int64)v10, (__int64)v20, 0, 0);
  if ( (unsigned __int8)sub_920620(v12) )
  {
    v14 = v29;
    if ( v28 )
      sub_B99FD0(v12, 3u, v28);
    sub_B45150(v12, v14);
  }
  (*(void (__fastcall **)(__int64, __int64, _WORD *, _QWORD, _QWORD))(*(_QWORD *)v27 + 16LL))(
    v27,
    v12,
    v19,
    v25[5],
    v25[6]);
  v15 = v23;
  v16 = &v23[2 * v24];
  if ( v23 != v16 )
  {
    do
    {
      v17 = v15[1];
      v18 = *(_DWORD *)v15;
      v15 += 2;
      sub_B99FD0(v12, v18, v17);
    }
    while ( v16 != v15 );
  }
LABEL_8:
  nullsub_61();
  v30 = &unk_49DA100;
  nullsub_63();
  if ( v23 != v25 )
    _libc_free((unsigned __int64)v23);
  nullsub_61();
  v22[14] = &unk_49DA100;
  nullsub_63();
  if ( (_QWORD *)v21[0] != v22 )
    _libc_free(v21[0]);
  return v12;
}
