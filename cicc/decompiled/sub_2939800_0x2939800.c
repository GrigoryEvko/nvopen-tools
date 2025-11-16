// Function: sub_2939800
// Address: 0x2939800
//
__int64 __fastcall sub_2939800(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  char v17; // bl
  __int64 v18; // r15
  __int64 v19; // rax
  char *v20; // rax
  unsigned int v21; // eax
  unsigned int v22; // r14d
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  _QWORD *v31; // rbx
  _QWORD *v32; // r15
  void (__fastcall *v33)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v34; // rax
  unsigned __int64 v36[2]; // [rsp+10h] [rbp-900h] BYREF
  _BYTE v37[512]; // [rsp+20h] [rbp-8F0h] BYREF
  __int64 v38; // [rsp+220h] [rbp-6F0h]
  __int64 v39; // [rsp+228h] [rbp-6E8h]
  __int64 v40; // [rsp+230h] [rbp-6E0h]
  __int64 v41; // [rsp+238h] [rbp-6D8h]
  char v42; // [rsp+240h] [rbp-6D0h]
  __int64 v43; // [rsp+248h] [rbp-6C8h]
  char *v44; // [rsp+250h] [rbp-6C0h]
  __int64 v45; // [rsp+258h] [rbp-6B8h]
  int v46; // [rsp+260h] [rbp-6B0h]
  char v47; // [rsp+264h] [rbp-6ACh]
  char v48; // [rsp+268h] [rbp-6A8h] BYREF
  __int16 v49; // [rsp+2A8h] [rbp-668h]
  _QWORD *v50; // [rsp+2B0h] [rbp-660h]
  _QWORD *v51; // [rsp+2B8h] [rbp-658h]
  __int64 v52; // [rsp+2C0h] [rbp-650h]
  _QWORD v53[4]; // [rsp+2D0h] [rbp-640h] BYREF
  char v54; // [rsp+2F0h] [rbp-620h]
  __int64 v55; // [rsp+2F8h] [rbp-618h]
  __int64 v56; // [rsp+300h] [rbp-610h]
  __int64 v57; // [rsp+308h] [rbp-608h]
  __int64 v58; // [rsp+310h] [rbp-600h]
  char *v59; // [rsp+318h] [rbp-5F8h]
  __int64 v60; // [rsp+320h] [rbp-5F0h]
  char v61; // [rsp+328h] [rbp-5E8h] BYREF
  char *v62; // [rsp+3A8h] [rbp-568h]
  __int64 v63; // [rsp+3B0h] [rbp-560h]
  char v64; // [rsp+3B8h] [rbp-558h] BYREF
  __int64 v65; // [rsp+478h] [rbp-498h]
  __int64 v66; // [rsp+480h] [rbp-490h]
  __int64 v67; // [rsp+488h] [rbp-488h]
  __int64 v68; // [rsp+490h] [rbp-480h]
  char *v69; // [rsp+498h] [rbp-478h]
  __int64 v70; // [rsp+4A0h] [rbp-470h]
  char v71; // [rsp+4A8h] [rbp-468h] BYREF
  __int64 v72; // [rsp+528h] [rbp-3E8h]
  char *v73; // [rsp+530h] [rbp-3E0h]
  __int64 v74; // [rsp+538h] [rbp-3D8h]
  int v75; // [rsp+540h] [rbp-3D0h]
  char v76; // [rsp+544h] [rbp-3CCh]
  char v77; // [rsp+548h] [rbp-3C8h] BYREF
  char *v78; // [rsp+5C8h] [rbp-348h]
  __int64 v79; // [rsp+5D0h] [rbp-340h]
  char v80; // [rsp+5D8h] [rbp-338h] BYREF
  __int64 v81; // [rsp+608h] [rbp-308h]
  __int64 v82; // [rsp+610h] [rbp-300h]
  __int64 v83; // [rsp+618h] [rbp-2F8h]
  __int64 v84; // [rsp+620h] [rbp-2F0h]
  char *v85; // [rsp+628h] [rbp-2E8h]
  __int64 v86; // [rsp+630h] [rbp-2E0h]
  char v87; // [rsp+638h] [rbp-2D8h] BYREF
  __int64 v88; // [rsp+678h] [rbp-298h]
  __int64 v89; // [rsp+680h] [rbp-290h]
  char v90; // [rsp+688h] [rbp-288h] BYREF
  _QWORD v91[2]; // [rsp+708h] [rbp-208h] BYREF
  char v92; // [rsp+718h] [rbp-1F8h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_37:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8144C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_37;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8144C);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 176;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F8662C )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_35;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F8662C);
  v11 = sub_CFFAC0(v10, a2);
  v12 = *(__int64 **)(a1 + 8);
  v38 = 0;
  v13 = v11;
  v39 = 0;
  v36[0] = (unsigned __int64)v37;
  v36[1] = 0x1000000000LL;
  v44 = &v48;
  v40 = v7;
  v41 = 0;
  v42 = 1;
  v43 = 0;
  v45 = 8;
  v46 = 0;
  v47 = 1;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F86530 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_36;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F86530);
  v17 = *(_BYTE *)(a1 + 169);
  v18 = *(_QWORD *)(v16 + 176);
  v19 = sub_B2BE50(a2);
  v54 = v17;
  v53[0] = v19;
  v60 = 0x1000000000LL;
  v70 = 0x1000000000LL;
  v59 = &v61;
  v73 = &v77;
  v62 = &v64;
  v78 = &v80;
  v53[2] = v18;
  v63 = 0x800000000LL;
  v85 = &v87;
  v86 = 0x800000000LL;
  v20 = &v90;
  v53[1] = v36;
  v53[3] = v13;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = &v71;
  v72 = 0;
  v74 = 16;
  v75 = 0;
  v76 = 1;
  v79 = 0x600000000LL;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v88 = 0;
  v89 = 1;
  do
  {
    *(_QWORD *)v20 = -4096;
    v20 += 16;
  }
  while ( v20 != (char *)v91 );
  v91[0] = &v92;
  v91[1] = 0x800000000LL;
  LOWORD(v21) = sub_29385B0((__int64)v53, a2);
  v22 = v21;
  sub_2913FD0((__int64)v53);
  sub_FFCE90((__int64)v36, a2, v23, v24, v25, v26);
  sub_FFD870((__int64)v36, a2, v27, v28, v29, v30);
  sub_FFBC40((__int64)v36, a2);
  v31 = v51;
  v32 = v50;
  if ( v51 != v50 )
  {
    do
    {
      v33 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v32[7];
      *v32 = &unk_49E5048;
      if ( v33 )
        v33(v32 + 5, v32 + 5, 3);
      *v32 = &unk_49DB368;
      v34 = v32[3];
      if ( v34 != 0 && v34 != -4096 && v34 != -8192 )
        sub_BD60C0(v32 + 1);
      v32 += 9;
    }
    while ( v31 != v32 );
    v32 = v50;
  }
  if ( v32 )
    j_j___libc_free_0((unsigned __int64)v32);
  if ( !v47 )
    _libc_free((unsigned __int64)v44);
  if ( (_BYTE *)v36[0] != v37 )
    _libc_free(v36[0]);
  return v22;
}
