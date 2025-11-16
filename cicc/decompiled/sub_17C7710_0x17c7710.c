// Function: sub_17C7710
// Address: 0x17c7710
//
__int64 __fastcall sub_17C7710(__int64 a1)
{
  __int64 v1; // rax
  int v2; // eax
  unsigned int v3; // r14d
  __int64 v5; // rax
  __int64 *v6; // r14
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // r14
  char v10; // al
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // r15
  char *v14; // rax
  size_t v15; // rdx
  __int64 *v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // r15
  __int64 v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // r15
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdx
  unsigned __int8 *v26; // rsi
  _QWORD *v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 *v30; // r15
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 v34; // rsi
  unsigned __int8 *v35; // rsi
  _BYTE *v36; // rsi
  __int64 v37; // [rsp+8h] [rbp-C8h]
  __int64 v38; // [rsp+10h] [rbp-C0h]
  int v39; // [rsp+10h] [rbp-C0h]
  __int64 v40; // [rsp+10h] [rbp-C0h]
  int v41; // [rsp+18h] [rbp-B8h]
  _QWORD *v42; // [rsp+18h] [rbp-B8h]
  unsigned __int64 *v43; // [rsp+18h] [rbp-B8h]
  __int64 v44; // [rsp+18h] [rbp-B8h]
  _QWORD *v45; // [rsp+18h] [rbp-B8h]
  __int64 v46; // [rsp+18h] [rbp-B8h]
  __int64 v47; // [rsp+18h] [rbp-B8h]
  __int64 v48; // [rsp+28h] [rbp-A8h] BYREF
  const char *v49; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v50; // [rsp+38h] [rbp-98h]
  __int16 v51; // [rsp+40h] [rbp-90h]
  const char **v52; // [rsp+50h] [rbp-80h] BYREF
  __int64 v53; // [rsp+58h] [rbp-78h]
  unsigned __int64 *v54; // [rsp+60h] [rbp-70h] BYREF
  __int64 v55; // [rsp+68h] [rbp-68h]
  __int64 v56; // [rsp+70h] [rbp-60h]
  int v57; // [rsp+78h] [rbp-58h]
  int v58; // [rsp+7Ch] [rbp-54h]
  __int64 v59; // [rsp+80h] [rbp-50h]
  __int64 v60; // [rsp+88h] [rbp-48h]

  v1 = *(_QWORD *)(a1 + 40);
  v51 = 260;
  v49 = (const char *)(v1 + 240);
  sub_16E1010((__int64)&v52, (__int64)&v49);
  v2 = v58;
  if ( v52 != (const char **)&v54 )
  {
    v41 = v58;
    j_j___libc_free_0(v52, (char *)v54 + 1);
    v2 = v41;
  }
  v3 = 0;
  if ( v2 != 9 && !sub_16321C0(*(_QWORD *)(a1 + 40), (__int64)"__llvm_profile_runtime", 22, 0) )
  {
    v5 = sub_1643350(**(_QWORD ***)(a1 + 40));
    v49 = "__llvm_profile_runtime";
    v6 = (__int64 *)v5;
    v50 = 22;
    LOWORD(v54) = 261;
    v52 = &v49;
    v42 = sub_1648A60(88, 1u);
    if ( v42 )
      sub_15E51E0((__int64)v42, *(_QWORD *)(a1 + 40), (__int64)v6, 0, 0, 0, (__int64)&v52, 0, 0, 0, 0);
    v7 = *(_QWORD *)(a1 + 40);
    v52 = &v49;
    LOWORD(v54) = 261;
    v37 = v7;
    v49 = "__llvm_profile_runtime_user";
    v50 = 27;
    v38 = sub_16453E0(v6, 0);
    v8 = sub_1648B60(120);
    v9 = v8;
    if ( v8 )
      sub_15E2490(v8, v38, 3, (__int64)&v52, v37);
    sub_15E0D50(v9, -1, 26);
    if ( *(_BYTE *)a1 )
      sub_15E0D50(v9, -1, 28);
    v10 = *(_BYTE *)(v9 + 32) & 0xCF | 0x10;
    *(_BYTE *)(v9 + 32) = v10;
    if ( (v10 & 0xF) != 9 )
      *(_BYTE *)(v9 + 33) |= 0x40u;
    v11 = *(_QWORD *)(a1 + 40);
    v51 = 260;
    v49 = (const char *)(v11 + 240);
    sub_16E1010((__int64)&v52, (__int64)&v49);
    v12 = HIDWORD(v59);
    if ( v52 != (const char **)&v54 )
    {
      v39 = HIDWORD(v59);
      j_j___libc_free_0(v52, (char *)v54 + 1);
      v12 = v39;
    }
    if ( v12 != 3 )
    {
      v13 = *(_QWORD *)(a1 + 40);
      v14 = (char *)sub_1649960(v9);
      *(_QWORD *)(v9 + 48) = sub_1633B90(v13, v14, v15);
    }
    v16 = *(__int64 **)(a1 + 40);
    v51 = 257;
    v40 = *v16;
    v17 = (_QWORD *)sub_22077B0(64);
    v18 = (__int64)v17;
    if ( v17 )
      sub_157FB60(v17, v40, (__int64)&v49, v9, 0);
    v19 = sub_157E9C0(v18);
    v53 = v18;
    v54 = (unsigned __int64 *)(v18 + 40);
    v52 = 0;
    v55 = v19;
    v56 = 0;
    v57 = 0;
    v59 = 0;
    v60 = 0;
    v51 = 257;
    v20 = sub_1648A60(64, 1u);
    v21 = v20;
    if ( v20 )
      sub_15F9210((__int64)v20, *(_QWORD *)(*v42 + 24LL), (__int64)v42, 0, 0, 0);
    if ( v53 )
    {
      v43 = v54;
      sub_157E9D0(v53 + 40, (__int64)v21);
      v22 = *v43;
      v23 = v21[3] & 7LL;
      v21[4] = v43;
      v22 &= 0xFFFFFFFFFFFFFFF8LL;
      v21[3] = v22 | v23;
      *(_QWORD *)(v22 + 8) = v21 + 3;
      *v43 = *v43 & 7 | (unsigned __int64)(v21 + 3);
    }
    sub_164B780((__int64)v21, (__int64 *)&v49);
    if ( v52 )
    {
      v48 = (__int64)v52;
      sub_1623A60((__int64)&v48, (__int64)v52, 2);
      v24 = v21[6];
      v25 = (__int64)(v21 + 6);
      if ( v24 )
      {
        sub_161E7C0((__int64)(v21 + 6), v24);
        v25 = (__int64)(v21 + 6);
      }
      v26 = (unsigned __int8 *)v48;
      v21[6] = v48;
      if ( v26 )
        sub_1623210((__int64)&v48, v26, v25);
    }
    v51 = 257;
    v44 = v55;
    v27 = sub_1648A60(56, v21 != 0);
    v28 = (__int64)v27;
    if ( v27 )
    {
      v29 = v44;
      v45 = v27;
      sub_15F6F90((__int64)v27, v29, (__int64)v21, 0);
      v28 = (__int64)v45;
    }
    if ( v53 )
    {
      v30 = (__int64 *)v54;
      v46 = v28;
      sub_157E9D0(v53 + 40, v28);
      v28 = v46;
      v31 = *v30;
      v32 = *(_QWORD *)(v46 + 24);
      *(_QWORD *)(v46 + 32) = v30;
      v31 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v46 + 24) = v31 | v32 & 7;
      *(_QWORD *)(v31 + 8) = v46 + 24;
      *v30 = *v30 & 7 | (v46 + 24);
    }
    v47 = v28;
    sub_164B780(v28, (__int64 *)&v49);
    if ( v52 )
    {
      v48 = (__int64)v52;
      sub_1623A60((__int64)&v48, (__int64)v52, 2);
      v33 = v47;
      v34 = *(_QWORD *)(v47 + 48);
      if ( v34 )
      {
        sub_161E7C0(v47 + 48, v34);
        v33 = v47;
      }
      v35 = (unsigned __int8 *)v48;
      *(_QWORD *)(v33 + 48) = v48;
      if ( v35 )
        sub_1623210((__int64)&v48, v35, v47 + 48);
    }
    v49 = (const char *)v9;
    v36 = *(_BYTE **)(a1 + 152);
    if ( v36 == *(_BYTE **)(a1 + 160) )
    {
      sub_167C6C0(a1 + 144, v36, &v49);
    }
    else
    {
      if ( v36 )
      {
        *(_QWORD *)v36 = v9;
        v36 = *(_BYTE **)(a1 + 152);
      }
      *(_QWORD *)(a1 + 152) = v36 + 8;
    }
    v3 = 1;
    if ( v52 )
      sub_161E7C0((__int64)&v52, (__int64)v52);
  }
  return v3;
}
