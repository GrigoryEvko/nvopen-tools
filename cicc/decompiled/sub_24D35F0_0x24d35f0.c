// Function: sub_24D35F0
// Address: 0x24d35f0
//
__int64 __fastcall sub_24D35F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  _BYTE *v4; // rax
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rax
  _QWORD *v8; // rax
  __int64 v9; // r14
  _BYTE *v10; // rbx
  unsigned __int64 v11; // r12
  __int64 v12; // rdx
  unsigned int v13; // esi
  char v15; // [rsp-138h] [rbp-138h]
  const char *v16; // [rsp-128h] [rbp-128h] BYREF
  char v17; // [rsp-108h] [rbp-108h]
  char v18; // [rsp-107h] [rbp-107h]
  _WORD v19[24]; // [rsp-F8h] [rbp-F8h] BYREF
  _BYTE *v20; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v21; // [rsp-C0h] [rbp-C0h]
  _BYTE v22[32]; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v23; // [rsp-98h] [rbp-98h]
  __int64 v24; // [rsp-90h] [rbp-90h]
  __int64 v25; // [rsp-88h] [rbp-88h]
  __int64 v26; // [rsp-80h] [rbp-80h]
  void **v27; // [rsp-78h] [rbp-78h]
  void **v28; // [rsp-70h] [rbp-70h]
  __int64 v29; // [rsp-68h] [rbp-68h]
  int v30; // [rsp-60h] [rbp-60h]
  __int16 v31; // [rsp-5Ch] [rbp-5Ch]
  char v32; // [rsp-5Ah] [rbp-5Ah]
  __int64 v33; // [rsp-58h] [rbp-58h]
  __int64 v34; // [rsp-50h] [rbp-50h]
  void *v35; // [rsp-48h] [rbp-48h] BYREF
  void *v36; // [rsp-40h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  if ( !v2 )
    BUG();
  v3 = *(_QWORD *)(v2 + 32);
  if ( v3 )
    v3 -= 24;
  v27 = &v35;
  v26 = sub_BD5C60(v3);
  v20 = v22;
  v35 = &unk_49DA100;
  v21 = 0x200000000LL;
  LOWORD(v25) = 0;
  v31 = 512;
  v36 = &unk_49DA0B0;
  v28 = &v36;
  v29 = 0;
  v30 = 0;
  v32 = 7;
  v33 = 0;
  v34 = 0;
  v23 = 0;
  v24 = 0;
  sub_D5F1F0((__int64)&v20, v3);
  v4 = sub_BA8D60(*(_QWORD *)(a2 + 40), (__int64)"__tysan_shadow_memory_address", 0x1Du, *(_QWORD *)(a1 + 72));
  v5 = *(_QWORD *)(a1 + 72);
  v18 = 1;
  v6 = (__int64)v4;
  v17 = 3;
  v16 = "shadow.base";
  v7 = sub_AA4E30(v23);
  v19[16] = 257;
  v15 = sub_AE5020(v7, v5);
  v8 = sub_BD2C40(80, unk_3F10A14);
  v9 = (__int64)v8;
  if ( v8 )
    sub_B4D190((__int64)v8, v5, v6, (__int64)v19, 0, v15, 0, 0);
  (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v28 + 2))(v28, v9, &v16, v24, v25);
  v10 = &v20[16 * (unsigned int)v21];
  if ( v20 != v10 )
  {
    v11 = (unsigned __int64)v20;
    do
    {
      v12 = *(_QWORD *)(v11 + 8);
      v13 = *(_DWORD *)v11;
      v11 += 16LL;
      sub_B99FD0(v9, v13, v12);
    }
    while ( v10 != (_BYTE *)v11 );
  }
  nullsub_61();
  v35 = &unk_49DA100;
  nullsub_63();
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
  return v9;
}
