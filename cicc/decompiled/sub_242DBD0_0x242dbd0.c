// Function: sub_242DBD0
// Address: 0x242dbd0
//
void __fastcall sub_242DBD0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r12
  _QWORD *v9; // rax
  _QWORD *v10; // rdi
  __int64 *v11; // rax
  __int64 *v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  int v15; // edx
  _QWORD *v16; // r15
  _QWORD *v17; // rax
  __int64 v18; // r9
  __int64 v19; // rbx
  unsigned int *v20; // r12
  unsigned int *v21; // r15
  __int64 v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // [rsp-10h] [rbp-140h]
  __int64 v25; // [rsp+20h] [rbp-110h]
  __int64 v26; // [rsp+28h] [rbp-108h]
  _QWORD v27[2]; // [rsp+30h] [rbp-100h] BYREF
  _QWORD v28[4]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v29; // [rsp+60h] [rbp-D0h]
  unsigned int *v30; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v31; // [rsp+78h] [rbp-B8h]
  _BYTE v32[32]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v33; // [rsp+A0h] [rbp-90h]
  __int64 v34; // [rsp+A8h] [rbp-88h]
  __int64 v35; // [rsp+B0h] [rbp-80h]
  _QWORD *v36; // [rsp+B8h] [rbp-78h]
  void **v37; // [rsp+C0h] [rbp-70h]
  void **v38; // [rsp+C8h] [rbp-68h]
  __int64 v39; // [rsp+D0h] [rbp-60h]
  int v40; // [rsp+D8h] [rbp-58h]
  __int16 v41; // [rsp+DCh] [rbp-54h]
  char v42; // [rsp+DEh] [rbp-52h]
  __int64 v43; // [rsp+E0h] [rbp-50h]
  __int64 v44; // [rsp+E8h] [rbp-48h]
  void *v45; // [rsp+F0h] [rbp-40h] BYREF
  void *v46; // [rsp+F8h] [rbp-38h] BYREF

  v26 = sub_242AEB0(a1, *(_QWORD *)a2, *(_DWORD *)(a2 + 8));
  v25 = sub_2427F80(a1, *(__int64 **)a2, *(unsigned int *)(a2 + 8));
  v3 = (__int64 *)sub_BCB120(*(_QWORD **)(a1 + 168));
  v4 = sub_BCF640(v3, 0);
  v5 = sub_2425400(a1, v4, (__int64)"__llvm_gcov_init", 16, (__int64)"_ZTSFvvE", 8);
  sub_B2CD30(v5, 31);
  v32[17] = 1;
  v30 = (unsigned int *)"entry";
  v6 = *(_QWORD *)(a1 + 168);
  v32[16] = 3;
  v7 = sub_22077B0(0x50u);
  v8 = v7;
  if ( v7 )
    sub_AA4D50(v7, v6, (__int64)&v30, v5, 0);
  v9 = (_QWORD *)sub_AA48A0(v8);
  v10 = *(_QWORD **)(a1 + 168);
  v36 = v9;
  v37 = &v45;
  v38 = &v46;
  v30 = (unsigned int *)v32;
  v45 = &unk_49DA100;
  v31 = 0x200000000LL;
  v33 = v8;
  LOWORD(v35) = 0;
  v41 = 512;
  v34 = v8 + 48;
  v39 = 0;
  v40 = 0;
  v42 = 7;
  v43 = 0;
  v44 = 0;
  v46 = &unk_49DA0B0;
  v11 = (__int64 *)sub_BCB120(v10);
  sub_BCF640(v11, 0);
  v28[0] = sub_BCE3C0(*(__int64 **)(a1 + 168), 0);
  v28[1] = v28[0];
  v12 = (__int64 *)sub_BCB120(v36);
  v13 = sub_BCF480(v12, v28, 2, 0);
  v14 = sub_BA8CA0(*(_QWORD *)(a1 + 128), (__int64)"llvm_gcov_init", 0xEu, v13);
  v29 = 257;
  v27[0] = v26;
  v27[1] = v25;
  sub_921880(&v30, v14, v15, (int)v27, 2, (__int64)v28, 0);
  v16 = v36;
  v29 = 257;
  v17 = sub_BD2C40(72, 0);
  v18 = v24;
  v19 = (__int64)v17;
  if ( v17 )
    sub_B4BB80((__int64)v17, (__int64)v16, 0, 0, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64, __int64))*v38 + 2))(
    v38,
    v19,
    v28,
    v34,
    v35,
    v18);
  v20 = &v30[4 * (unsigned int)v31];
  if ( v30 != v20 )
  {
    v21 = v30;
    do
    {
      v22 = *((_QWORD *)v21 + 1);
      v23 = *v21;
      v21 += 4;
      sub_B99FD0(v19, v23, v22);
    }
    while ( v20 != v21 );
  }
  sub_2A3ED40(*(_QWORD *)(a1 + 128), v5, 0, 0);
  nullsub_61();
  v45 = &unk_49DA100;
  nullsub_63();
  if ( v30 != (unsigned int *)v32 )
    _libc_free((unsigned __int64)v30);
}
