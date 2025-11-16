// Function: sub_370B390
// Address: 0x370b390
//
__int64 __fastcall sub_370B390(_QWORD *a1, __int16 *a2)
{
  __int16 v2; // ax
  _WORD *v3; // rax
  __int16 v4; // ax
  __int64 v5; // r14
  volatile signed __int32 *v6; // r12
  signed __int32 v7; // eax
  void (*v8)(); // rax
  signed __int32 v9; // eax
  __int64 (__fastcall *v11)(__int64); // rdx
  _WORD *v12; // [rsp+8h] [rbp-108h]
  _WORD v13[2]; // [rsp+24h] [rbp-ECh] BYREF
  __int64 v14; // [rsp+28h] [rbp-E8h] BYREF
  unsigned __int64 v15; // [rsp+30h] [rbp-E0h] BYREF
  unsigned __int64 v16; // [rsp+38h] [rbp-D8h]
  _QWORD v17[2]; // [rsp+40h] [rbp-D0h] BYREF
  volatile signed __int32 *v18; // [rsp+50h] [rbp-C0h]
  __int64 v19; // [rsp+78h] [rbp-98h]
  void *v20; // [rsp+80h] [rbp-90h] BYREF
  char v21; // [rsp+8Ah] [rbp-86h]
  char v22; // [rsp+8Eh] [rbp-82h]
  _BYTE *v23; // [rsp+90h] [rbp-80h]
  __int64 v24; // [rsp+98h] [rbp-78h]
  _BYTE v25[24]; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v26; // [rsp+B8h] [rbp-58h]
  _QWORD *v27; // [rsp+C0h] [rbp-50h]
  __int64 v28; // [rsp+C8h] [rbp-48h]
  __int64 v29; // [rsp+D0h] [rbp-40h]

  sub_3719220(v17, *a1, a1[1] - *a1, 1);
  v13[0] = 2;
  v21 = 0;
  v22 = 0;
  v20 = &unk_4A3C998;
  v23 = v25;
  v24 = 0x200000000LL;
  v2 = *a2;
  v26 = 0;
  v27 = v17;
  v28 = 0;
  v29 = 0;
  v13[1] = v2;
  sub_3719260(&v15, v17, v13, 4);
  if ( (v15 & 0xFFFFFFFFFFFFFFFELL) != 0
    || (v3 = (_WORD *)*a1,
        v16 = 4,
        v12 = v3,
        v15 = (unsigned __int64)v3,
        sub_370EAB0(&v14, &v20, &v15),
        (v14 & 0xFFFFFFFFFFFFFFFELL) != 0)
    || (sub_370D0A0(&v14, &v20), (v14 & 0xFFFFFFFFFFFFFFFELL) != 0)
    || (sub_370CE40(&v14, &v20, &v15), (v14 & 0xFFFFFFFFFFFFFFFELL) != 0) )
  {
    BUG();
  }
  sub_3708E80((__int64)v17);
  v4 = 0;
  if ( v16 > 3 )
    v4 = *(_WORD *)(v15 + 2);
  v12[1] = v4;
  *v12 = v19 - 2;
  v5 = *a1;
  if ( v23 != v25 )
    _libc_free((unsigned __int64)v23);
  v6 = v18;
  v17[0] = &unk_4A352E0;
  if ( v18 )
  {
    if ( &_pthread_key_create )
    {
      v7 = _InterlockedExchangeAdd(v18 + 2, 0xFFFFFFFF);
    }
    else
    {
      v7 = *((_DWORD *)v18 + 2);
      *((_DWORD *)v18 + 2) = v7 - 1;
    }
    if ( v7 == 1 )
    {
      v8 = *(void (**)())(*(_QWORD *)v6 + 16LL);
      if ( v8 != nullsub_25 )
        ((void (__fastcall *)(volatile signed __int32 *))v8)(v6);
      if ( &_pthread_key_create )
      {
        v9 = _InterlockedExchangeAdd(v6 + 3, 0xFFFFFFFF);
      }
      else
      {
        v9 = *((_DWORD *)v6 + 3);
        *((_DWORD *)v6 + 3) = v9 - 1;
      }
      if ( v9 == 1 )
      {
        v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 24LL);
        if ( v11 == sub_9C26E0 )
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v6 + 8LL))(v6);
        else
          v11((__int64)v6);
      }
    }
  }
  return v5;
}
