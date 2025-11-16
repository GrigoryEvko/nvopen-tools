// Function: sub_2587130
// Address: 0x2587130
//
char __fastcall sub_2587130(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 (__fastcall *v7)(__int64); // rax
  __int64 v8; // rdi
  __int64 v9; // r8
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rsi
  void (__fastcall *v12)(__int64, unsigned __int64, unsigned __int64); // rax
  _QWORD *v13; // rdi
  __int64 (*v14)(void); // rax
  __int64 v16; // [rsp-10h] [rbp-20h]

  v3 = sub_250D2C0(a2, **(_QWORD **)a1);
  v5 = sub_2584D90(*(_QWORD *)(a1 + 8), v3, v4, *(_QWORD *)(a1 + 16), 0, 0, 1);
  if ( !v5 )
    return 0;
  v6 = v5;
  v7 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 48LL);
  if ( v7 == sub_2534F40 )
    v8 = v6 + 88;
  else
    v8 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64))v7)(v6, v3, v16);
  v9 = *(_QWORD *)(a1 + 24);
  if ( !*(_BYTE *)(v9 + 24) )
  {
    *(_QWORD *)(v9 + 8) = 1;
    *(_QWORD *)(v9 + 16) = 0x100000000LL;
    *(_BYTE *)(v9 + 24) = 1;
    *(_QWORD *)v9 = &unk_4A16ED8;
    v9 = *(_QWORD *)(a1 + 24);
  }
  v10 = *(_QWORD *)(v8 + 8);
  v11 = *(_QWORD *)(v8 + 16);
  v12 = *(void (__fastcall **)(__int64, unsigned __int64, unsigned __int64))(*(_QWORD *)v9 + 72LL);
  if ( v12 == sub_2535490 )
  {
    if ( *(_QWORD *)(v9 + 8) <= v10 )
      v10 = *(_QWORD *)(v9 + 8);
    if ( *(_QWORD *)(v9 + 16) <= v11 )
      v11 = *(_QWORD *)(v9 + 16);
    *(_QWORD *)(v9 + 8) = v10;
    *(_QWORD *)(v9 + 16) = v11;
  }
  else
  {
    v12(v9, v11, v10);
  }
  v13 = *(_QWORD **)(a1 + 24);
  v14 = *(__int64 (**)(void))(*v13 + 16LL);
  if ( (char *)v14 == (char *)sub_2505FD0 )
    return v13[2] != 1;
  else
    return v14();
}
