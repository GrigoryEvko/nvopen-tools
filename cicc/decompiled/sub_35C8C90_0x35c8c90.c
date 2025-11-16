// Function: sub_35C8C90
// Address: 0x35c8c90
//
__int64 __fastcall sub_35C8C90(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  unsigned __int8 v6; // al
  __int64 *v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int8 v12; // al
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int8 *v17; // r15
  unsigned __int8 *v18; // rax
  bool v19; // cc
  _QWORD *v20; // rax
  __int64 v21; // rax
  __int64 v22; // r9
  unsigned int v23; // edx
  unsigned __int64 v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // rax
  char v28; // cl
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // r12
  __int64 *v33; // rax
  __int64 v34; // rdx
  unsigned __int8 *v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r15
  __int64 v39; // r14
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // [rsp-150h] [rbp-150h]
  unsigned int v44; // [rsp-140h] [rbp-140h]
  __int64 v45; // [rsp-130h] [rbp-130h] BYREF
  unsigned __int64 v46; // [rsp-128h] [rbp-128h] BYREF
  unsigned int v47; // [rsp-120h] [rbp-120h]
  _QWORD v48[2]; // [rsp-118h] [rbp-118h] BYREF
  _QWORD *v49; // [rsp-108h] [rbp-108h]
  __int64 v50; // [rsp-100h] [rbp-100h]
  _QWORD v51[6]; // [rsp-F8h] [rbp-F8h] BYREF
  _QWORD *v52; // [rsp-C8h] [rbp-C8h]
  __int64 v53; // [rsp-C0h] [rbp-C0h]
  _QWORD v54[6]; // [rsp-B8h] [rbp-B8h] BYREF
  __int16 v55; // [rsp-88h] [rbp-88h]
  __int64 *v56; // [rsp-80h] [rbp-80h]
  void **v57; // [rsp-78h] [rbp-78h]
  void **v58; // [rsp-70h] [rbp-70h]
  __int64 v59; // [rsp-68h] [rbp-68h]
  int v60; // [rsp-60h] [rbp-60h]
  __int16 v61; // [rsp-5Ch] [rbp-5Ch]
  char v62; // [rsp-5Ah] [rbp-5Ah]
  __int64 v63; // [rsp-58h] [rbp-58h]
  __int64 v64; // [rsp-50h] [rbp-50h]
  void *v65; // [rsp-48h] [rbp-48h] BYREF
  void *v66; // [rsp-40h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(*a2 + 7) & 0x20) != 0 )
  {
    v3 = sub_B91C10(*a2, 37);
    v4 = v3;
    if ( !v3 )
      return 0;
    v5 = v3 - 16;
    v6 = *(_BYTE *)(v3 - 16);
    v7 = (v6 & 2) != 0 ? *(__int64 **)(v4 - 32) : (__int64 *)(v5 - 8LL * ((v6 >> 2) & 0xF));
    v8 = *v7;
    v9 = sub_B91420(*v7);
    if ( v10 <= 0xC
      || *(_QWORD *)v9 != 0x6F635F646D6E6173LL
      || *(_DWORD *)(v9 + 8) != 1701995894
      || *(_BYTE *)(v9 + 12) != 100 )
    {
      return 0;
    }
    v12 = *(_BYTE *)(v4 - 16);
    if ( (v12 & 2) != 0 )
      v13 = *(_QWORD *)(v4 - 32);
    else
      v13 = v5 - 8LL * ((v12 >> 2) & 0xF);
    v14 = *(_QWORD *)(v13 + 8);
    v15 = *(unsigned __int8 *)(v14 - 16);
    if ( (v15 & 2) != 0 )
    {
      v16 = *(_QWORD *)(v14 - 32);
    }
    else
    {
      v15 = 8LL * (((unsigned __int8)v15 >> 2) & 0xF);
      v16 = v14 - 16 - v15;
    }
    v17 = *(unsigned __int8 **)(*(_QWORD *)v16 + 136LL);
    v18 = sub_AD8340(v17, 37, v15);
    v19 = *((_DWORD *)v18 + 2) <= 0x40u;
    v20 = *(_QWORD **)v18;
    if ( !v19 )
      v20 = (_QWORD *)*v20;
    if ( ((unsigned __int8)v20 & 2) == 0 )
      return 0;
    v21 = a2[6];
    if ( *(int *)(v21 + 32) <= 0 )
      return 0;
    v22 = *(_QWORD *)(v21 + 8);
    v23 = *(_DWORD *)(v21 + 32) - 1;
    v24 = 0;
    v25 = 0;
    do
    {
      v26 = v22 + 40LL * v23;
      v27 = *(_QWORD *)(v26 + 8) + *(_QWORD *)v26;
      v28 = *(_BYTE *)(v26 + 16);
      if ( v25 < v27 )
        v25 = v27;
      if ( v24 < 1LL << v28 )
        v24 = 1LL << v28;
    }
    while ( v23-- != 0 );
    v30 = v24 + v25 - 1;
    v31 = -(__int64)v24;
    v44 = v31 & v30;
    if ( (v31 & v30) == 0 )
      return 0;
    v32 = *a2;
    v33 = (__int64 *)sub_B2BE50(*a2);
    v57 = &v65;
    v56 = v33;
    v61 = 512;
    v52 = v54;
    v65 = &unk_49DA100;
    v53 = 0x200000000LL;
    v55 = 0;
    v58 = &v66;
    v59 = 0;
    v60 = 0;
    v62 = 7;
    v63 = 0;
    v64 = 0;
    v54[4] = 0;
    v54[5] = 0;
    v66 = &unk_49DA0B0;
    v45 = sub_B2BE50(v32);
    v35 = sub_AD8340(v17, v31, v34);
    v47 = *((_DWORD *)v35 + 2);
    if ( v47 > 0x40 )
    {
      sub_C43780((__int64)&v46, (const void **)v35);
      if ( v47 > 0x40 )
      {
        *(_QWORD *)v46 |= 4uLL;
        goto LABEL_30;
      }
    }
    else
    {
      v46 = *(_QWORD *)v35;
    }
    v46 |= 4u;
LABEL_30:
    v36 = sub_B91420(v8);
    v38 = v37;
    v39 = v36;
    v43 = sub_ACCFD0(v56, (__int64)&v46);
    v40 = sub_BCB2D0(v56);
    v41 = sub_ACD640(v40, v44, 0);
    v48[0] = v39;
    v51[1] = v41;
    v51[0] = v43;
    v48[1] = v38;
    v49 = v51;
    v50 = 0x600000002LL;
    v42 = sub_B8CB30(&v45, (__int64)v48, 1);
    sub_B99110(v32, 37, v42);
    if ( v49 != v51 )
      _libc_free((unsigned __int64)v49);
    if ( v47 > 0x40 && v46 )
      j_j___libc_free_0_0(v46);
    nullsub_61();
    v65 = &unk_49DA100;
    nullsub_63();
    if ( v52 != v54 )
      _libc_free((unsigned __int64)v52);
    return 0;
  }
  return 0;
}
