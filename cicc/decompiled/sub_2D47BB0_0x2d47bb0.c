// Function: sub_2D47BB0
// Address: 0x2d47bb0
//
__int64 __fastcall sub_2D47BB0(unsigned __int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rdi
  __int64 v5; // r13
  void (*v6)(); // rax
  unsigned __int64 v8[2]; // [rsp+0h] [rbp-130h] BYREF
  char v9; // [rsp+10h] [rbp-120h] BYREF
  void *v10; // [rsp+80h] [rbp-B0h]
  void *v11; // [rsp+88h] [rbp-A8h]
  _QWORD v12[10]; // [rsp+E0h] [rbp-50h] BYREF

  sub_2D46B10((__int64)v8, a2, a1[1]);
  v3 = (*(__int64 (__fastcall **)(unsigned __int64, unsigned __int64 *, _QWORD, _QWORD, _QWORD))(*(_QWORD *)*a1 + 1032LL))(
         *a1,
         v8,
         *(_QWORD *)(a2 + 8),
         *(_QWORD *)(a2 - 32),
         (*(_WORD *)(a2 + 2) >> 7) & 7);
  v4 = *a1;
  v5 = v3;
  v6 = *(void (**)())(*(_QWORD *)*a1 + 1120LL);
  if ( v6 != nullsub_1577 )
    ((void (__fastcall *)(unsigned __int64, unsigned __int64 *))v6)(v4, v8);
  sub_BD84D0(a2, v5);
  sub_B43D60((_QWORD *)a2);
  sub_B32BF0(v12);
  v10 = &unk_49E5698;
  v11 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( (char *)v8[0] != &v9 )
    _libc_free(v8[0]);
  return 1;
}
