// Function: sub_2D48EC0
// Address: 0x2d48ec0
//
void __fastcall sub_2D48EC0(
        _QWORD *a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __int64 (__fastcall *a7)(__int64, __int64, __int64),
        __int64 a8)
{
  __int64 v11; // rax
  unsigned __int64 v12[2]; // [rsp+10h] [rbp-140h] BYREF
  char v13; // [rsp+20h] [rbp-130h] BYREF
  void *v14; // [rsp+90h] [rbp-C0h]
  void *v15; // [rsp+98h] [rbp-B8h]
  _QWORD v16[12]; // [rsp+F0h] [rbp-60h] BYREF

  sub_2D46B10((__int64)v12, (__int64)a2, a1[1]);
  v11 = sub_2D46690(a1, (__int64)v12, a3, a4, a5, (__int64)v12, a7, a8);
  sub_BD84D0((__int64)a2, v11);
  sub_B43D60(a2);
  sub_B32BF0(v16);
  v14 = &unk_49E5698;
  v15 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( (char *)v12[0] != &v13 )
    _libc_free(v12[0]);
}
