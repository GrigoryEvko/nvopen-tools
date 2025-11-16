// Function: sub_3010A20
// Address: 0x3010a20
//
__int64 __fastcall sub_3010A20(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 *v5; // rax
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  int v12; // eax
  __int128 v13; // rax
  _BYTE *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 *v21; // rdi
  __int64 v22; // r12
  __int64 *v23; // rax
  unsigned __int64 v24; // rax
  __int64 v25; // r14
  _BYTE *v26; // rdx
  _BYTE *v27; // r12
  __int64 *v28; // r12
  unsigned int v29; // r13d
  unsigned int v30; // ecx
  __int64 v31; // r14
  __int64 v32; // rax
  __int64 *v33; // rdi
  __int64 *v34; // r12
  __int64 *v35; // r13
  __int64 v36; // rsi
  unsigned int v37; // r12d
  __int64 v39; // rax
  __int64 v40; // r8
  __int64 v41; // r9
  char v42; // al
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  __int64 *v47; // [rsp+38h] [rbp-278h]
  __int64 *v48; // [rsp+38h] [rbp-278h]
  _OWORD v49[2]; // [rsp+40h] [rbp-270h] BYREF
  __int64 v50; // [rsp+60h] [rbp-250h]
  const char *v51; // [rsp+70h] [rbp-240h]
  __int64 v52; // [rsp+78h] [rbp-238h]
  char v53; // [rsp+90h] [rbp-220h]
  char v54; // [rsp+91h] [rbp-21Fh]
  unsigned __int64 v55[2]; // [rsp+A0h] [rbp-210h] BYREF
  _QWORD v56[2]; // [rsp+B0h] [rbp-200h] BYREF
  __int64 v57; // [rsp+C0h] [rbp-1F0h]
  __int64 v58[2]; // [rsp+D0h] [rbp-1E0h] BYREF
  _BYTE v59[32]; // [rsp+E0h] [rbp-1D0h] BYREF
  __int64 v60; // [rsp+100h] [rbp-1B0h]
  __int64 v61; // [rsp+108h] [rbp-1A8h]
  __int16 v62; // [rsp+110h] [rbp-1A0h]
  __int64 *v63; // [rsp+118h] [rbp-198h]
  void **v64; // [rsp+120h] [rbp-190h]
  void **v65; // [rsp+128h] [rbp-188h]
  __int64 v66; // [rsp+130h] [rbp-180h]
  int v67; // [rsp+138h] [rbp-178h]
  __int16 v68; // [rsp+13Ch] [rbp-174h]
  char v69; // [rsp+13Eh] [rbp-172h]
  __int64 v70; // [rsp+140h] [rbp-170h]
  __int64 v71; // [rsp+148h] [rbp-168h]
  void *v72; // [rsp+150h] [rbp-160h] BYREF
  void *v73; // [rsp+158h] [rbp-158h] BYREF
  __int64 *v74; // [rsp+160h] [rbp-150h] BYREF
  __int64 v75; // [rsp+168h] [rbp-148h]
  _BYTE v76[128]; // [rsp+170h] [rbp-140h] BYREF
  __int64 *v77; // [rsp+1F0h] [rbp-C0h] BYREF
  __int64 v78; // [rsp+1F8h] [rbp-B8h]
  _BYTE v79[176]; // [rsp+200h] [rbp-B0h] BYREF

  v3 = a2 + 72;
  v47 = *(__int64 **)(a2 + 40);
  v5 = (__int64 *)sub_B2BE50(a2);
  v6 = *(_QWORD *)(a2 + 80);
  v63 = v5;
  v64 = &v72;
  v65 = &v73;
  v58[0] = (__int64)v59;
  v72 = &unk_49DA100;
  v58[1] = 0x200000000LL;
  v68 = 512;
  v73 = &unk_49DA0B0;
  v74 = (__int64 *)v76;
  v66 = 0;
  v67 = 0;
  v69 = 7;
  v70 = 0;
  v71 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v75 = 0x1000000000LL;
  v77 = (__int64 *)v79;
  v78 = 0x1000000000LL;
  if ( v6 == a2 + 72 )
    goto LABEL_45;
  do
  {
    v7 = v6 - 24;
    if ( !v6 )
      v7 = 0;
    v8 = sub_AA4FF0(v7);
    if ( !v8 )
      BUG();
    v9 = (unsigned int)*(unsigned __int8 *)(v8 - 24) - 39;
    if ( (unsigned int)v9 <= 0x38 )
    {
      v10 = 0x100060000000001LL;
      if ( _bittest64(&v10, v9) )
      {
        v39 = sub_AA4FF0(v7);
        if ( !v39 )
          BUG();
        v42 = *(_BYTE *)(v39 - 24);
        if ( v42 == 81 )
        {
          v45 = (unsigned int)v75;
          v46 = (unsigned int)v75 + 1LL;
          if ( v46 > HIDWORD(v75) )
          {
            sub_C8D5F0((__int64)&v74, v76, v46, 8u, v40, v41);
            v45 = (unsigned int)v75;
          }
          v74[v45] = v7;
          LODWORD(v75) = v75 + 1;
        }
        else if ( v42 == 80 )
        {
          v43 = (unsigned int)v78;
          v44 = (unsigned int)v78 + 1LL;
          if ( v44 > HIDWORD(v78) )
          {
            sub_C8D5F0((__int64)&v77, v79, v44, 8u, v40, v41);
            v43 = (unsigned int)v78;
          }
          v77[v43] = v7;
          LODWORD(v78) = v78 + 1;
        }
      }
    }
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( v3 != v6 );
  if ( !(_DWORD)v75 && !(_DWORD)v78 )
  {
LABEL_45:
    v33 = v77;
    v37 = 0;
    goto LABEL_28;
  }
  if ( (*(_BYTE *)(a2 + 2) & 8) == 0 )
    goto LABEL_12;
  v11 = sub_B2E500(a2);
  v12 = sub_B2A630(v11);
  if ( v12 > 10 )
  {
    if ( v12 == 12 )
      goto LABEL_14;
LABEL_12:
    v54 = 1;
    v51 = "' does not have a correct Wasm personality function '__gxx_wasm_personality_v0'";
    v53 = 3;
    *(_QWORD *)&v13 = sub_BD5D20(a2);
    *(_QWORD *)&v49[0] = "Function '";
    v49[1] = v13;
    LOWORD(v50) = 1283;
    v56[0] = "' does not have a correct Wasm personality function '__gxx_wasm_personality_v0'";
    v55[0] = (unsigned __int64)v49;
    v56[1] = v52;
    LOWORD(v57) = 770;
    sub_C64D30((__int64)v55, 1u);
  }
  if ( v12 <= 6 )
    goto LABEL_12;
LABEL_14:
  v14 = sub_BA8D60((__int64)v47, (__int64)"__wasm_lpad_context", 0x13u, *a1);
  a1[1] = (__int64)v14;
  v14[33] = v14[33] & 0xE3 | 4;
  v15 = a1[1];
  v16 = *a1;
  BYTE1(v57) = 1;
  a1[2] = v15;
  v55[0] = (unsigned __int64)"lsda_gep";
  LOBYTE(v57) = 3;
  v17 = sub_24DBB60(v58, v16, v15, 0, 1u, (__int64)v55);
  v18 = a1[1];
  v19 = *a1;
  a1[3] = v17;
  v55[0] = (unsigned __int64)"selector_gep";
  LOWORD(v57) = 259;
  a1[4] = sub_24DBB60(v58, v19, v18, 0, 2u, (__int64)v55);
  a1[6] = sub_B6E160(v47, 0x3774u, 0, 0);
  a1[7] = sub_B6E160(v47, 0x3776u, 0, 0);
  a1[8] = sub_B6E160(v47, 0x3773u, 0, 0);
  a1[10] = sub_B6E160(v47, 0x3772u, 0, 0);
  v20 = sub_B6E160(v47, 0x376Du, 0, 0);
  v21 = v63;
  a1[9] = v20;
  v22 = sub_BCE3C0(v21, 0);
  v23 = (__int64 *)sub_BCB2D0(v63);
  v56[0] = v22;
  v55[0] = (unsigned __int64)v56;
  v55[1] = 0x100000001LL;
  v24 = sub_BCF480(v23, v56, 1, 0);
  v25 = sub_BA8C10((__int64)v47, (__int64)"_Unwind_CallPersonality", 0x17u, v24, 0);
  v27 = v26;
  if ( (_QWORD *)v55[0] != v56 )
    _libc_free(v55[0]);
  a1[11] = v25;
  a1[12] = (__int64)v27;
  if ( !*v27 )
    sub_B2CD30((__int64)v27, 41);
  v28 = v74;
  v29 = 0;
  v48 = &v74[(unsigned int)v75];
  if ( v48 != v74 )
  {
    do
    {
      while ( 1 )
      {
        v31 = *v28;
        v32 = sub_AA4FF0(*v28);
        if ( !v32 )
          BUG();
        if ( (*(_DWORD *)(v32 - 20) & 0x7FFFFFF) == 2 && sub_AC30F0(*(_QWORD *)(v32 - 88)) )
          break;
        v30 = v29++;
        ++v28;
        sub_300FDC0(a1, v31, 1, v30);
        if ( v48 == v28 )
          goto LABEL_25;
      }
      sub_300FDC0(a1, v31, 0, 0);
      ++v28;
    }
    while ( v48 != v28 );
  }
LABEL_25:
  v33 = v77;
  v34 = &v77[(unsigned int)v78];
  v35 = v77;
  if ( v34 == v77 )
  {
    v37 = 1;
  }
  else
  {
    do
    {
      v36 = *v35++;
      sub_300FDC0(a1, v36, 0, 0);
    }
    while ( v34 != v35 );
    v33 = v77;
    v37 = 1;
  }
LABEL_28:
  if ( v33 != (__int64 *)v79 )
    _libc_free((unsigned __int64)v33);
  if ( v74 != (__int64 *)v76 )
    _libc_free((unsigned __int64)v74);
  nullsub_61();
  v72 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v58[0] != v59 )
    _libc_free(v58[0]);
  return v37;
}
