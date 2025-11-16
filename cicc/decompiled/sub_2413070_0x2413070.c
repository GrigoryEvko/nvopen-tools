// Function: sub_2413070
// Address: 0x2413070
//
void __fastcall sub_2413070(__int64 *a1, unsigned __int64 a2, int a3, char a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r12
  _QWORD *v14; // rax
  __int64 v15; // r9
  __int64 v16; // r13
  _QWORD *v17; // rbx
  _QWORD *v18; // r12
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int16 v21; // [rsp-100h] [rbp-100h]
  _WORD v22[24]; // [rsp-F8h] [rbp-F8h] BYREF
  _QWORD *v23; // [rsp-C8h] [rbp-C8h] BYREF
  unsigned int v24; // [rsp-C0h] [rbp-C0h]
  _QWORD v25[9]; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v26; // [rsp-70h] [rbp-70h]
  void *v27; // [rsp-48h] [rbp-48h]

  if ( !a5 )
    BUG();
  v21 = a6;
  sub_2412230((__int64)&v23, *(_QWORD *)(a5 + 16), a5, a6, 0, a6, 0, 0);
  v9 = sub_BCCE00(*(_QWORD **)(*a1 + 8), 8 * a3);
  v10 = sub_ACD640(v9, 0, 0);
  v13 = sub_2412E00(*a1, a2, a5, v21, v11, v12);
  v22[16] = 257;
  v14 = sub_BD2C40(80, unk_3F10A10);
  v16 = (__int64)v14;
  if ( v14 )
    sub_B4D3C0((__int64)v14, v10, v13, 0, a4, v15, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _WORD *, _QWORD, _QWORD))(*(_QWORD *)v26 + 16LL))(
    v26,
    v16,
    v22,
    v25[5],
    v25[6]);
  v17 = v23;
  v18 = &v23[2 * v24];
  if ( v23 != v18 )
  {
    do
    {
      v19 = v17[1];
      v20 = *(_DWORD *)v17;
      v17 += 2;
      sub_B99FD0(v16, v20, v19);
    }
    while ( v18 != v17 );
  }
  nullsub_61();
  v27 = &unk_49DA100;
  nullsub_63();
  if ( v23 != v25 )
    _libc_free((unsigned __int64)v23);
}
