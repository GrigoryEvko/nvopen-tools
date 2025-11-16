// Function: sub_396D030
// Address: 0x396d030
//
void __fastcall sub_396D030(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  const char *v4; // rsi
  unsigned int v5; // r15d
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 v10; // rdx
  __int64 v11; // rdi
  void (*v12)(); // rcx
  unsigned __int64 v13; // rdx
  void (*v14)(void); // rax
  _QWORD *v15; // [rsp+8h] [rbp-128h]
  _QWORD v16[2]; // [rsp+10h] [rbp-120h] BYREF
  __int64 v17[2]; // [rsp+20h] [rbp-110h] BYREF
  void (__fastcall *v18)(__int64 *, __int64 *, __int64); // [rsp+30h] [rbp-100h]
  void (__fastcall *v19)(__int64 *, _QWORD *); // [rsp+38h] [rbp-F8h]
  _QWORD v20[2]; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v21; // [rsp+50h] [rbp-E0h]
  __int64 v22; // [rsp+58h] [rbp-D8h]
  int v23; // [rsp+60h] [rbp-D0h]
  unsigned __int64 *v24; // [rsp+68h] [rbp-C8h]
  unsigned __int64 v25[2]; // [rsp+70h] [rbp-C0h] BYREF
  _BYTE v26[176]; // [rsp+80h] [rbp-B0h] BYREF

  v3 = *(_QWORD *)(a2 + 32);
  v4 = 0;
  v25[0] = (unsigned __int64)v26;
  v5 = *(_DWORD *)(v3 + 8);
  v25[1] = 0x8000000000LL;
  v20[0] = &unk_49EFC48;
  v24 = v25;
  v23 = 1;
  v22 = 0;
  v21 = 0;
  v20[1] = 0;
  sub_16E7A40((__int64)v20, 0, 0, 0);
  v6 = v22;
  if ( (unsigned __int64)(v21 - v22) <= 0xD )
  {
    v4 = "implicit-def: ";
    v15 = (_QWORD *)sub_16E7EE0((__int64)v20, "implicit-def: ", 0xEu);
  }
  else
  {
    *(_DWORD *)(v22 + 8) = 1717920813;
    *(_QWORD *)v6 = 0x746963696C706D69LL;
    *(_WORD *)(v6 + 12) = 8250;
    v22 += 14;
    v15 = v20;
  }
  v7 = 0;
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 264) + 16LL);
  v9 = *(__int64 (**)())(*(_QWORD *)v8 + 112LL);
  if ( v9 != sub_1D00B10 )
    v7 = ((__int64 (__fastcall *)(__int64, const char *, _QWORD))v9)(v8, v4, 0);
  sub_1F4AA00(v17, v5, v7, 0, 0);
  if ( !v18 )
    sub_4263D6(v17, v5, v10);
  v19(v17, v15);
  if ( v18 )
    v18(v17, v17, 3);
  v11 = *(_QWORD *)(a1 + 256);
  v12 = *(void (**)())(*(_QWORD *)v11 + 104LL);
  v13 = *v24;
  v16[1] = *((unsigned int *)v24 + 2);
  LOWORD(v18) = 261;
  v16[0] = v13;
  v17[0] = (__int64)v16;
  if ( v12 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, __int64 *, __int64))v12)(v11, v17, 1);
    v11 = *(_QWORD *)(a1 + 256);
  }
  v14 = *(void (**)(void))(*(_QWORD *)v11 + 144LL);
  if ( v14 != nullsub_581 )
    v14();
  v20[0] = &unk_49EFD28;
  sub_16E7960((__int64)v20);
  if ( (_BYTE *)v25[0] != v26 )
    _libc_free(v25[0]);
}
