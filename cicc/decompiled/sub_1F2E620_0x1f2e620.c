// Function: sub_1F2E620
// Address: 0x1f2e620
//
__int64 __fastcall sub_1F2E620(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  _DWORD *v9; // rbx
  __int64 v10; // rdi
  __int64 (*v11)(); // rax
  __int64 v12; // rdi
  __int64 (*v13)(); // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v18; // rax
  __int64 v19; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v20[5]; // [rsp+8h] [rbp-28h] BYREF

  *(_QWORD *)(a1 + 232) = a2;
  *(_QWORD *)(a1 + 240) = *(_QWORD *)(a2 + 40);
  v3 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9E06C, 1u);
  if ( v3 && (v4 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v3 + 104LL))(v3, &unk_4F9E06C)) != 0 )
    v5 = v4 + 160;
  else
    v5 = 0;
  v6 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 248) = v5;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_22:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4FCBA30 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_22;
  }
  v9 = *(_DWORD **)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(
                      *(_QWORD *)(v7 + 8),
                      &unk_4FCBA30)
                  + 208);
  *(_QWORD *)(a1 + 160) = v9;
  sub_2240AE0(a1 + 176, v9 + 118);
  v10 = *(_QWORD *)(a1 + 160);
  *(_DWORD *)(a1 + 208) = v9[126];
  *(_DWORD *)(a1 + 212) = v9[127];
  *(_DWORD *)(a1 + 216) = v9[128];
  *(_DWORD *)(a1 + 220) = v9[129];
  *(_DWORD *)(a1 + 224) = v9[130];
  *(_DWORD *)(a1 + 228) = v9[131];
  v11 = *(__int64 (**)())(*(_QWORD *)v10 + 16LL);
  if ( v11 == sub_16FF750 )
    BUG();
  v12 = ((__int64 (__fastcall *)(__int64, __int64))v11)(v10, a2);
  v13 = *(__int64 (**)())(*(_QWORD *)v12 + 56LL);
  v14 = 0;
  if ( v13 != sub_1D12D20 )
    v14 = ((__int64 (__fastcall *)(__int64))v13)(v12);
  *(_QWORD *)(a1 + 168) = v14;
  *(_WORD *)(a1 + 464) = 0;
  v19 = sub_1560340((_QWORD *)(a2 + 112), -1, "stack-protector-buffer-size", 0x1Bu);
  if ( sub_155D3E0((__int64)&v19) )
  {
    v15 = sub_155D8B0(&v19);
    if ( sub_16D2B80(v15, v16, 0xAu, v20) || v20[0] != LODWORD(v20[0]) )
      return 0;
    *(_DWORD *)(a1 + 288) = v20[0];
  }
  if ( !(unsigned __int8)sub_1F2C610(a1) )
    return 0;
  if ( (*(_BYTE *)(a2 + 18) & 8) != 0 )
  {
    v18 = sub_15E38F0(a2);
    if ( (unsigned int)sub_14DD7D0(v18) - 7 <= 3 )
      return 0;
  }
  return sub_1F2D8B0(a1);
}
