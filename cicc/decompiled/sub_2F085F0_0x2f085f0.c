// Function: sub_2F085F0
// Address: 0x2f085f0
//
void __fastcall sub_2F085F0(__int64 a1, _BYTE *a2)
{
  char v2; // bl
  __int64 v4; // rax
  __int64 v5; // rax
  const char *v6; // rdx
  void (__fastcall *v7)(__int64, unsigned __int64 *); // rcx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  char v11; // al
  unsigned __int64 v12; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-110h] BYREF
  void *v14; // [rsp+10h] [rbp-100h] BYREF
  __int64 v15; // [rsp+18h] [rbp-F8h]
  __int64 v16; // [rsp+20h] [rbp-F0h]
  __int64 v17; // [rsp+28h] [rbp-E8h]
  __int64 v18; // [rsp+30h] [rbp-E0h]
  __int64 v19; // [rsp+38h] [rbp-D8h]
  unsigned __int64 *v20; // [rsp+40h] [rbp-D0h]
  char *v21; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v22; // [rsp+58h] [rbp-B8h]
  __int64 v23; // [rsp+60h] [rbp-B0h]
  char v24; // [rsp+68h] [rbp-A8h] BYREF
  __int16 v25; // [rsp+70h] [rbp-A0h]

  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v4 = *(_QWORD *)a1;
    v14 = 0;
    v15 = 0;
    (*(void (__fastcall **)(__int64, void **, _QWORD))(v4 + 216))(a1, &v14, 0);
    sub_CB0A70(a1);
    if ( sub_C93C90((__int64)v14, v15, 0xAu, (unsigned __int64 *)&v21) )
    {
      v5 = 14;
      v6 = "invalid number";
LABEL_4:
      v7 = *(void (__fastcall **)(__int64, unsigned __int64 *))(*(_QWORD *)a1 + 248LL);
      v25 = 261;
      v21 = (char *)v6;
      v22 = v5;
      v7(a1, (unsigned __int64 *)&v21);
      return;
    }
    if ( v21 )
    {
      if ( ((unsigned __int64)v21 & (unsigned __int64)(v21 - 1)) != 0 )
      {
        v5 = 27;
        v6 = "must be 0 or a power of two";
        goto LABEL_4;
      }
      _BitScanReverse64(&v12, (unsigned __int64)v21);
      v2 = 63 - (v12 ^ 0x3F);
      v11 = 1;
    }
    else
    {
      v11 = 0;
    }
    *a2 = v2;
    a2[1] = v11;
    return;
  }
  v19 = 0x100000000LL;
  v20 = (unsigned __int64 *)&v21;
  v14 = &unk_49DD288;
  v21 = &v24;
  v22 = 0;
  v23 = 128;
  v15 = 2;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  sub_CB5980((__int64)&v14, 0, 0, 0);
  sub_CB0A70(a1);
  v8 = 0;
  if ( a2[1] )
    v8 = 1LL << *a2;
  sub_CB59D0((__int64)&v14, v8);
  v9 = v20[1];
  v13[0] = *v20;
  v10 = *(_QWORD *)a1;
  v13[1] = v9;
  (*(void (__fastcall **)(__int64, _QWORD *, _QWORD))(v10 + 216))(a1, v13, 0);
  v14 = &unk_49DD388;
  sub_CB5840((__int64)&v14);
  if ( v21 != &v24 )
    _libc_free((unsigned __int64)v21);
}
