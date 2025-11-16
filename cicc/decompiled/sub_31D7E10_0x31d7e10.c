// Function: sub_31D7E10
// Address: 0x31d7e10
//
void __fastcall sub_31D7E10(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // r15d
  __int64 v5; // rdx
  _QWORD *v6; // r9
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdi
  void (*v10)(); // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  void (*v13)(void); // rax
  _QWORD *v14; // [rsp+8h] [rbp-148h]
  __int64 v15[2]; // [rsp+10h] [rbp-140h] BYREF
  void (__fastcall *v16)(__int64 *, __int64 *, __int64); // [rsp+20h] [rbp-130h]
  void (__fastcall *v17)(__int64 *, _QWORD *); // [rsp+28h] [rbp-128h]
  __int16 v18; // [rsp+30h] [rbp-120h]
  _QWORD v19[3]; // [rsp+40h] [rbp-110h] BYREF
  __int64 v20; // [rsp+58h] [rbp-F8h]
  __int64 v21; // [rsp+60h] [rbp-F0h]
  __int64 v22; // [rsp+68h] [rbp-E8h]
  __int64 *v23; // [rsp+70h] [rbp-E0h]
  unsigned __int64 v24[3]; // [rsp+80h] [rbp-D0h] BYREF
  _BYTE v25[184]; // [rsp+98h] [rbp-B8h] BYREF

  v3 = *(_QWORD *)(a2 + 32);
  v24[0] = (unsigned __int64)v25;
  v4 = *(_DWORD *)(v3 + 8);
  v22 = 0x100000000LL;
  v19[0] = &unk_49DD288;
  v23 = (__int64 *)v24;
  v24[1] = 0;
  v24[2] = 128;
  v19[1] = 2;
  v19[2] = 0;
  v20 = 0;
  v21 = 0;
  sub_CB5980((__int64)v19, 0, 0, 0);
  v5 = v21;
  if ( (unsigned __int64)(v20 - v21) <= 0xD )
  {
    v6 = (_QWORD *)sub_CB6200((__int64)v19, "implicit-def: ", 0xEu);
  }
  else
  {
    *(_DWORD *)(v21 + 8) = 1717920813;
    v6 = v19;
    *(_QWORD *)v5 = 0x746963696C706D69LL;
    *(_WORD *)(v5 + 12) = 8250;
    v21 += 14;
  }
  v14 = v6;
  v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 232) + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)(a1 + 232) + 16LL));
  sub_2FF6320(v15, v4, v7, 0, 0);
  if ( !v16 )
    sub_4263D6(v15, v4, v8);
  v17(v15, v14);
  if ( v16 )
    v16(v15, v15, 3);
  v9 = *(_QWORD *)(a1 + 224);
  v10 = *(void (**)())(*(_QWORD *)v9 + 120LL);
  v11 = v23[1];
  v12 = *v23;
  v18 = 261;
  v15[0] = v12;
  v15[1] = v11;
  if ( v10 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, __int64 *, __int64))v10)(v9, v15, 1);
    v9 = *(_QWORD *)(a1 + 224);
  }
  v13 = *(void (**)(void))(*(_QWORD *)v9 + 160LL);
  if ( v13 != nullsub_99 )
    v13();
  v19[0] = &unk_49DD388;
  sub_CB5840((__int64)v19);
  if ( (_BYTE *)v24[0] != v25 )
    _libc_free(v24[0]);
}
