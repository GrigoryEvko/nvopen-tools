// Function: sub_2CC0040
// Address: 0x2cc0040
//
__int64 __fastcall sub_2CC0040(__int64 a1, unsigned __int8 a2, __int64 a3, __int16 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int8 *v7; // r14
  __int64 (__fastcall *v8)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char); // rax
  __int64 v9; // r12
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  __int64 v13; // rdx
  unsigned int v14; // esi
  void *v15; // [rsp-118h] [rbp-118h] BYREF
  char v16; // [rsp-F8h] [rbp-F8h]
  char v17; // [rsp-F7h] [rbp-F7h]
  _WORD v18[24]; // [rsp-E8h] [rbp-E8h] BYREF
  _QWORD *v19; // [rsp-B8h] [rbp-B8h] BYREF
  unsigned int v20; // [rsp-B0h] [rbp-B0h]
  _QWORD v21[8]; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v22; // [rsp-68h] [rbp-68h]
  __int64 v23; // [rsp-60h] [rbp-60h]
  void *v24; // [rsp-38h] [rbp-38h]

  if ( !a3 )
    BUG();
  sub_2412230((__int64)&v19, *(_QWORD *)(a3 + 16), a3, a4, 0, a6, 0, 0);
  v6 = sub_AD64C0(*(_QWORD *)(a1 + 8), 1, a2);
  v17 = 1;
  v7 = (unsigned __int8 *)v6;
  v16 = 3;
  v15 = &unk_42D2000;
  v8 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *, unsigned __int8, char))(*(_QWORD *)v22 + 32LL);
  if ( v8 != sub_9201A0 )
  {
    v9 = ((__int64 (__fastcall *)(__int64, __int64, __int64, unsigned __int8 *, _QWORD, _QWORD, void *))v8)(
           v22,
           13,
           a1,
           v7,
           0,
           0,
           v15);
    goto LABEL_7;
  }
  if ( *(_BYTE *)a1 <= 0x15u && *v7 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(13) )
      v9 = sub_AD5570(13, a1, v7, 0, 0);
    else
      v9 = sub_AABE40(0xDu, (unsigned __int8 *)a1, v7);
LABEL_7:
    if ( v9 )
      goto LABEL_8;
  }
  v18[16] = 257;
  v9 = sub_B504D0(13, a1, (__int64)v7, (__int64)v18, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, void **, _QWORD, _QWORD))(*(_QWORD *)v23 + 16LL))(
    v23,
    v9,
    &v15,
    v21[5],
    v21[6]);
  v11 = v19;
  v12 = &v19[2 * v20];
  if ( v19 != v12 )
  {
    do
    {
      v13 = v11[1];
      v14 = *(_DWORD *)v11;
      v11 += 2;
      sub_B99FD0(v9, v14, v13);
    }
    while ( v12 != v11 );
  }
LABEL_8:
  nullsub_61();
  v24 = &unk_49DA100;
  nullsub_63();
  if ( v19 != v21 )
    _libc_free((unsigned __int64)v19);
  return v9;
}
