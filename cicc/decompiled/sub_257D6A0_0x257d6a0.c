// Function: sub_257D6A0
// Address: 0x257d6a0
//
char __fastcall sub_257D6A0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 (__fastcall *v8)(__int64); // rax
  __int64 v9; // rdi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // rdx
  void (__fastcall *v14)(__int64, int, int); // rax
  _DWORD *v15; // rdi
  bool (__fastcall *v16)(__int64); // rax
  __int64 v18; // [rsp-10h] [rbp-20h]
  __int64 v19; // [rsp-8h] [rbp-18h]

  v3 = sub_250D2C0(a2, **(_QWORD **)a1);
  v5 = sub_257C550(*(_QWORD *)(a1 + 8), v3, v4, *(_QWORD *)(a1 + 16), 0, 0, 1);
  v6 = v19;
  if ( !v5 )
    return 0;
  v7 = v5;
  v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 48LL);
  if ( v8 == sub_2534FF0 )
    v9 = v7 + 88;
  else
    v9 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64))v8)(v7, v3, v18);
  v10 = *(_QWORD *)(a1 + 24);
  if ( !*(_BYTE *)(v10 + 16) )
  {
    *(_BYTE *)(v10 + 16) = 1;
    *(_QWORD *)v10 = &unk_4A172B8;
    *(_QWORD *)(v10 + 8) = 0x3FF00000000LL;
    v10 = *(_QWORD *)(a1 + 24);
  }
  v11 = *(unsigned int *)(v9 + 8);
  v12 = *(unsigned int *)(v9 + 12);
  v13 = *(_QWORD *)(v9 + 8);
  v14 = *(void (__fastcall **)(__int64, int, int))(*(_QWORD *)v10 + 72LL);
  if ( v14 == sub_2535320 )
    *(_QWORD *)(v10 + 8) &= v13;
  else
    v14(v10, v12, v11);
  v15 = *(_DWORD **)(a1 + 24);
  v16 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v15 + 16LL);
  if ( v16 == sub_2506010 )
    return v15[3] != 0;
  else
    return ((__int64 (__fastcall *)(_DWORD *, __int64, __int64, __int64, __int64, __int64))v16)(
             v15,
             v12,
             v13,
             v6,
             v10,
             v11);
}
