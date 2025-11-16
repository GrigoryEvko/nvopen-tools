// Function: sub_23E0390
// Address: 0x23e0390
//
__int64 __fastcall sub_23E0390(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v5; // rsi
  __int64 *v6; // rdi
  _QWORD **v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rsi
  int v11; // edx
  __int64 v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // r13
  __int64 v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // r12
  unsigned int *v18; // r13
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int64 v21; // [rsp-148h] [rbp-148h]
  char v22; // [rsp-13Ch] [rbp-13Ch]
  int v23; // [rsp-130h] [rbp-130h]
  unsigned int *v24; // [rsp-130h] [rbp-130h]
  _QWORD v25[4]; // [rsp-128h] [rbp-128h] BYREF
  __int16 v26; // [rsp-108h] [rbp-108h]
  _QWORD v27[4]; // [rsp-F8h] [rbp-F8h] BYREF
  __int16 v28; // [rsp-D8h] [rbp-D8h]
  unsigned int *v29; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v30; // [rsp-C0h] [rbp-C0h]
  _BYTE v31[32]; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v32; // [rsp-98h] [rbp-98h]
  __int64 v33; // [rsp-90h] [rbp-90h]
  __int64 v34; // [rsp-88h] [rbp-88h]
  __int64 v35; // [rsp-80h] [rbp-80h]
  void **v36; // [rsp-78h] [rbp-78h]
  void **v37; // [rsp-70h] [rbp-70h]
  __int64 v38; // [rsp-68h] [rbp-68h]
  int v39; // [rsp-60h] [rbp-60h]
  __int16 v40; // [rsp-5Ch] [rbp-5Ch]
  char v41; // [rsp-5Ah] [rbp-5Ah]
  __int64 v42; // [rsp-58h] [rbp-58h]
  __int64 v43; // [rsp-50h] [rbp-50h]
  void *v44; // [rsp-48h] [rbp-48h] BYREF
  void *v45; // [rsp-40h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a1 + 128) != -1 )
    return 0;
  v2 = *(_QWORD *)(a2 + 80);
  if ( !v2 )
    BUG();
  v3 = *(_QWORD *)(v2 + 32);
  if ( v3 )
    v3 -= 24;
  v35 = sub_BD5C60(v3);
  v29 = (unsigned int *)v31;
  v44 = &unk_49DA100;
  v30 = 0x200000000LL;
  v36 = &v44;
  v37 = &v45;
  v38 = 0;
  v39 = 0;
  v40 = 512;
  v41 = 7;
  v42 = 0;
  v43 = 0;
  v32 = 0;
  v33 = 0;
  LOWORD(v34) = 0;
  v45 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v29, v3);
  if ( *(_BYTE *)(a1 + 137) )
  {
    v5 = *(_QWORD *)(a1 + 192);
    v6 = *(__int64 **)(a1 + 96);
    if ( (_BYTE)qword_4FE1CC8 )
    {
      v27[0] = *(_QWORD *)(v5 + 8);
      v8 = (_QWORD **)sub_BCF480(v6, v27, 1, 0);
      v9 = sub_B41A60(v8, (__int64)byte_3F871B3, 0, (__int64)"=r,0", 4, 0, 0, 0, 0);
      v10 = 0;
      v11 = v9;
      v28 = 259;
      v27[0] = ".asan.shadow";
      v25[0] = *(_QWORD *)(a1 + 192);
      if ( v9 )
      {
        v23 = v9;
        v12 = sub_B3B7D0(v9);
        v11 = v23;
        v10 = v12;
      }
      *(_QWORD *)(a1 + 1016) = sub_921880(&v29, v10, v11, (int)v25, 1, (__int64)v27, 0);
    }
    else
    {
      v27[0] = ".asan.shadow";
      v28 = 259;
      *(_QWORD *)(a1 + 1016) = sub_94BCF0(&v29, v5, (__int64)v6, (__int64)v27);
    }
  }
  else
  {
    v13 = sub_BA8D60(*(_QWORD *)(a2 + 40), (__int64)"__asan_shadow_memory_dynamic_address", 0x24u, *(_QWORD *)(a1 + 96));
    v14 = *(_QWORD *)(a1 + 96);
    v26 = 257;
    v21 = (__int64)v13;
    v15 = sub_AA4E30(v32);
    v28 = 257;
    v22 = sub_AE5020(v15, v14);
    v16 = sub_BD2C40(80, unk_3F10A14);
    v17 = (__int64)v16;
    if ( v16 )
      sub_B4D190((__int64)v16, v14, v21, (__int64)v27, 0, v22, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v37 + 2))(v37, v17, v25, v33, v34);
    v18 = v29;
    v24 = &v29[4 * (unsigned int)v30];
    if ( v29 != v24 )
    {
      do
      {
        v19 = *((_QWORD *)v18 + 1);
        v20 = *v18;
        v18 += 4;
        sub_B99FD0(v17, v20, v19);
      }
      while ( v24 != v18 );
    }
    *(_QWORD *)(a1 + 1016) = v17;
  }
  nullsub_61();
  v44 = &unk_49DA100;
  nullsub_63();
  if ( v29 != (unsigned int *)v31 )
    _libc_free((unsigned __int64)v29);
  return 1;
}
