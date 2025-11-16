// Function: sub_2427F80
// Address: 0x2427f80
//
__int64 __fastcall sub_2427F80(__int64 a1, __int64 *a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 *v6; // rax
  unsigned __int64 v7; // r14
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // r13
  char v13; // dl
  __int64 v14; // r15
  __int16 v15; // ax
  __int64 v16; // rax
  __int64 v17; // r15
  unsigned int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  char v22; // al
  __int64 v23; // rax
  _QWORD *v24; // r13
  __int64 v25; // r12
  _QWORD *v26; // rax
  __int64 v27; // rbx
  unsigned __int64 v28; // r13
  char *v29; // r12
  __int64 v30; // rdx
  unsigned int v31; // esi
  _QWORD *v33; // r12
  _QWORD *v34; // rax
  __int64 v35; // rbx
  unsigned __int64 v36; // r13
  char *v37; // r12
  __int64 v38; // rdx
  unsigned int v39; // esi
  __int64 v40; // [rsp+18h] [rbp-118h]
  __int64 *v41; // [rsp+20h] [rbp-110h]
  _QWORD *v42; // [rsp+28h] [rbp-108h]
  char v43; // [rsp+36h] [rbp-FAh]
  char v44; // [rsp+37h] [rbp-F9h]
  unsigned __int64 v45; // [rsp+38h] [rbp-F8h]
  _BYTE v46[32]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v47; // [rsp+60h] [rbp-D0h]
  char *v48; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v49; // [rsp+78h] [rbp-B8h]
  _BYTE v50[32]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v51; // [rsp+A0h] [rbp-90h]
  __int64 v52; // [rsp+A8h] [rbp-88h]
  __int64 v53; // [rsp+B0h] [rbp-80h]
  _QWORD *v54; // [rsp+B8h] [rbp-78h]
  void **v55; // [rsp+C0h] [rbp-70h]
  void **v56; // [rsp+C8h] [rbp-68h]
  __int64 v57; // [rsp+D0h] [rbp-60h]
  int v58; // [rsp+D8h] [rbp-58h]
  __int16 v59; // [rsp+DCh] [rbp-54h]
  char v60; // [rsp+DEh] [rbp-52h]
  __int64 v61; // [rsp+E0h] [rbp-50h]
  __int64 v62; // [rsp+E8h] [rbp-48h]
  void *v63; // [rsp+F0h] [rbp-40h] BYREF
  void *v64; // [rsp+F8h] [rbp-38h] BYREF

  v6 = (__int64 *)sub_BCB120(*(_QWORD **)(a1 + 168));
  v7 = sub_BCF640(v6, 0);
  v40 = (__int64)sub_BA8CB0(*(_QWORD *)(a1 + 128), (__int64)"__llvm_gcov_reset", 0x11u);
  if ( !v40 )
    v40 = sub_2425400(a1, v7, (__int64)"__llvm_gcov_reset", 17, (__int64)"_ZTSFvvE", 8);
  sub_B2CD30(v40, 31);
  v8 = *(_QWORD *)(a1 + 168);
  v50[17] = 1;
  v48 = "entry";
  v50[16] = 3;
  v9 = sub_22077B0(0x50u);
  v10 = v9;
  if ( v9 )
    sub_AA4D50(v9, v8, (__int64)&v48, v40, 0);
  v11 = 2 * a3;
  v54 = (_QWORD *)sub_AA48A0(v10);
  v55 = &v63;
  v56 = &v64;
  v48 = v50;
  v63 = &unk_49DA100;
  v49 = 0x200000000LL;
  LOWORD(v53) = 0;
  v64 = &unk_49DA0B0;
  v57 = 0;
  v58 = 0;
  v59 = 512;
  v60 = 7;
  v61 = 0;
  v62 = 0;
  v51 = v10;
  v52 = v10 + 48;
  v42 = (_QWORD *)sub_AA48A0(v10);
  v41 = &a2[v11];
  while ( v41 != a2 )
  {
    v12 = *a2;
    v13 = 0;
    v14 = *(_QWORD *)(*a2 + 24);
    v15 = (*(_WORD *)(*a2 + 34) >> 1) & 0x3F;
    if ( v15 )
    {
      v13 = 1;
      v43 = v15 - 1;
    }
    v44 = v13;
    a2 += 2;
    v45 = (*(_QWORD *)(v14 + 32) * (unsigned __int64)(unsigned int)sub_BCB060(*(_QWORD *)(v14 + 24))) >> 3;
    v16 = sub_BCB2B0(v42);
    LOBYTE(v3) = v43;
    v17 = sub_AD6530(v16, v45);
    v18 = v3;
    BYTE1(v18) = v44;
    v3 = v18;
    v19 = sub_BCB2E0(v54);
    v20 = sub_ACD640(v19, v45, 0);
    sub_B34240((__int64)&v48, v12, v17, v20, v3, 0, 0, 0, 0);
  }
  v21 = **(_QWORD **)(*(_QWORD *)(v40 + 24) + 16LL);
  v22 = *(_BYTE *)(v21 + 8);
  if ( v22 == 7 )
  {
    v33 = v54;
    v47 = 257;
    v34 = sub_BD2C40(72, 0);
    v35 = (__int64)v34;
    if ( v34 )
      sub_B4BB80((__int64)v34, (__int64)v33, 0, 0, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v56 + 2))(v56, v35, v46, v52, v53);
    v36 = (unsigned __int64)v48;
    v37 = &v48[16 * (unsigned int)v49];
    if ( v48 != v37 )
    {
      do
      {
        v38 = *(_QWORD *)(v36 + 8);
        v39 = *(_DWORD *)v36;
        v36 += 16LL;
        sub_B99FD0(v35, v39, v38);
      }
      while ( v37 != (char *)v36 );
    }
  }
  else
  {
    if ( v22 != 12 )
      sub_C64ED0("invalid return type for __llvm_gcov_reset", 1u);
    v23 = sub_AD64C0(v21, 0, 0);
    v24 = v54;
    v25 = v23;
    v47 = 257;
    v26 = sub_BD2C40(72, v23 != 0);
    v27 = (__int64)v26;
    if ( v26 )
      sub_B4BB80((__int64)v26, (__int64)v24, v25, v25 != 0, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v56 + 2))(v56, v27, v46, v52, v53);
    v28 = (unsigned __int64)v48;
    v29 = &v48[16 * (unsigned int)v49];
    if ( v48 != v29 )
    {
      do
      {
        v30 = *(_QWORD *)(v28 + 8);
        v31 = *(_DWORD *)v28;
        v28 += 16LL;
        sub_B99FD0(v27, v31, v30);
      }
      while ( v29 != (char *)v28 );
    }
  }
  nullsub_61();
  v63 = &unk_49DA100;
  nullsub_63();
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
  return v40;
}
