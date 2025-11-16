// Function: sub_24D2AC0
// Address: 0x24d2ac0
//
void __fastcall sub_24D2AC0(_QWORD *a1, __int64 **a2)
{
  __int64 *v4; // rdi
  __int64 *v5; // rsi
  unsigned __int64 v6; // rax
  __int64 v7; // rcx
  __int64 *v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // rdi
  __int64 *v13; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // [rsp+0h] [rbp-130h]
  __int64 v19; // [rsp+8h] [rbp-128h]
  __int64 v20; // [rsp+10h] [rbp-120h]
  __int64 v21; // [rsp+18h] [rbp-118h]
  __int64 v22; // [rsp+18h] [rbp-118h]
  __int64 v23; // [rsp+20h] [rbp-110h]
  __int64 v24; // [rsp+20h] [rbp-110h]
  __int64 v25; // [rsp+20h] [rbp-110h]
  __int64 v26; // [rsp+20h] [rbp-110h]
  unsigned __int64 v27; // [rsp+38h] [rbp-F8h] BYREF
  _QWORD *v28; // [rsp+40h] [rbp-F0h]
  __int64 v29; // [rsp+48h] [rbp-E8h]
  _QWORD v30[4]; // [rsp+50h] [rbp-E0h] BYREF
  _BYTE *v31; // [rsp+70h] [rbp-C0h]
  __int64 v32; // [rsp+78h] [rbp-B8h]
  _BYTE v33[32]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+A0h] [rbp-90h]
  __int64 v35; // [rsp+A8h] [rbp-88h]
  __int16 v36; // [rsp+B0h] [rbp-80h]
  __int64 *v37; // [rsp+B8h] [rbp-78h]
  void **v38; // [rsp+C0h] [rbp-70h]
  void **v39; // [rsp+C8h] [rbp-68h]
  __int64 v40; // [rsp+D0h] [rbp-60h]
  int v41; // [rsp+D8h] [rbp-58h]
  __int16 v42; // [rsp+DCh] [rbp-54h]
  char v43; // [rsp+DEh] [rbp-52h]
  __int64 v44; // [rsp+E0h] [rbp-50h]
  __int64 v45; // [rsp+E8h] [rbp-48h]
  void *v46; // [rsp+F0h] [rbp-40h] BYREF
  void *v47; // [rsp+F8h] [rbp-38h] BYREF

  v4 = *a2;
  v36 = 0;
  v31 = v33;
  v32 = 0x200000000LL;
  v42 = 512;
  v37 = v4;
  v38 = &v46;
  v46 = &unk_49DA100;
  v39 = &v47;
  v40 = 0;
  v41 = 0;
  v43 = 7;
  v44 = 0;
  v45 = 0;
  v34 = 0;
  v35 = 0;
  v47 = &unk_49DA0B0;
  a1[11] = sub_BCB2D0(v4);
  v5 = *a2;
  v27 = 0;
  v6 = sub_A7A090((__int64 *)&v27, v5, -1, 41);
  v7 = a1[11];
  v27 = v6;
  v18 = v7;
  v19 = sub_BCE3C0(v37, 0);
  v20 = a1[11];
  v21 = sub_BCE3C0(v37, 0);
  v8 = (__int64 *)sub_BCB120(v37);
  v28 = v30;
  v30[0] = v21;
  v30[2] = v19;
  v30[3] = v18;
  v23 = v27;
  v30[1] = v20;
  v29 = 0x400000004LL;
  v9 = sub_BCF480(v8, v30, 4, 0);
  v10 = sub_BA8C10((__int64)a2, (__int64)"__tysan_check", 0xDu, v9, v23);
  if ( v28 != v30 )
  {
    v22 = v11;
    v24 = v10;
    _libc_free((unsigned __int64)v28);
    v11 = v22;
    v10 = v24;
  }
  a1[13] = v11;
  v12 = v37;
  a1[12] = v10;
  v13 = (__int64 *)sub_BCB120(v12);
  v28 = v30;
  v25 = v27;
  v29 = 0;
  v14 = sub_BCF480(v13, v30, 0, 0);
  v15 = sub_BA8C10((__int64)a2, (__int64)"tysan.module_ctor", 0x11u, v14, v25);
  v17 = v16;
  if ( v28 != v30 )
  {
    v26 = v15;
    _libc_free((unsigned __int64)v28);
    v15 = v26;
  }
  a1[14] = v15;
  a1[15] = v17;
  nullsub_61();
  v46 = &unk_49DA100;
  nullsub_63();
  if ( v31 != v33 )
    _libc_free((unsigned __int64)v31);
}
