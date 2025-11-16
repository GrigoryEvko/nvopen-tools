// Function: sub_2DC37C0
// Address: 0x2dc37c0
//
__int64 __fastcall sub_2DC37C0(__int64 a1, unsigned int a2, _DWORD *a3)
{
  __int64 v5; // rax
  __int64 v6; // r12
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 **v11; // r15
  __int64 v12; // rax
  __int64 (__fastcall *v13)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v14; // rax
  unsigned __int8 *v15; // r10
  __int64 v16; // r11
  __int64 v17; // r12
  __int64 v18; // rdi
  __int64 (__fastcall *v19)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v20; // r14
  _BYTE *v21; // rsi
  __int64 v22; // rax
  __int64 v23; // r12
  unsigned int v24; // r14d
  _QWORD *v25; // rax
  __int64 **v26; // rax
  __int64 v27; // rax
  unsigned __int8 *v28; // rdx
  __int64 *v29; // rdi
  __int64 v30; // rax
  __int64 (__fastcall *v31)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v35; // rdx
  __int64 v36; // r14
  __int64 v37; // rdx
  unsigned int v38; // esi
  _QWORD *v39; // rax
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v44; // rdx
  int v45; // ecx
  int v46; // eax
  _QWORD *v47; // rdi
  __int64 *v48; // rax
  __int64 v49; // rax
  __int64 v50; // r10
  __int64 v51; // r11
  __int64 v52; // r15
  __int64 v53; // r14
  __int64 v54; // rdx
  unsigned int v55; // esi
  __int64 v56; // rax
  __int64 v57; // rax
  unsigned int v58; // r14d
  _QWORD *v59; // rax
  __int64 *v60; // rax
  __int64 *v61; // rdi
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 *v64; // rax
  __int64 *v65; // rdi
  __int64 v66; // rax
  __int64 v67; // rax
  _BYTE *v68; // rax
  unsigned __int8 *v69; // [rsp+8h] [rbp-108h]
  unsigned __int8 *v70; // [rsp+8h] [rbp-108h]
  __int64 v71; // [rsp+10h] [rbp-100h]
  unsigned __int8 *v72; // [rsp+18h] [rbp-F8h]
  __int64 v73; // [rsp+18h] [rbp-F8h]
  __int64 v74; // [rsp+18h] [rbp-F8h]
  unsigned __int8 *v75; // [rsp+18h] [rbp-F8h]
  unsigned int v76; // [rsp+20h] [rbp-F0h]
  unsigned __int8 *v77; // [rsp+20h] [rbp-F0h]
  __int64 v78; // [rsp+20h] [rbp-F0h]
  unsigned __int8 *v79; // [rsp+20h] [rbp-F0h]
  unsigned int v80; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v81; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v82; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v83; // [rsp+28h] [rbp-E8h]
  __int64 v84; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v85; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v86; // [rsp+40h] [rbp-D0h] BYREF
  _BYTE *v87; // [rsp+48h] [rbp-C8h]
  _BYTE *v88; // [rsp+50h] [rbp-C0h]
  __int64 *v89; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v90; // [rsp+68h] [rbp-A8h]
  __int64 v91; // [rsp+70h] [rbp-A0h]
  __int64 v92[4]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v93; // [rsp+A0h] [rbp-70h]
  __int64 *v94; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v95; // [rsp+B8h] [rbp-58h]
  __int64 v96; // [rsp+C0h] [rbp-50h]
  __int16 v97; // [rsp+D0h] [rbp-40h]

  v5 = (unsigned int)*a3;
  v6 = *(unsigned int *)(a1 + 280);
  v86 = 0;
  v87 = 0;
  v7 = v6 - v5;
  v8 = *(_QWORD *)(a1 + 56);
  v88 = 0;
  v89 = 0;
  v90 = 0;
  if ( v7 > v8 )
    LODWORD(v7) = v8;
  v71 = a1 + 128;
  v9 = *(_QWORD *)(a1 + 64);
  v91 = 0;
  v84 = 0;
  v76 = v7;
  if ( v9 == *(_QWORD *)(a1 + 72) )
  {
    sub_D5F1F0(a1 + 128, *(_QWORD *)a1);
  }
  else
  {
    v10 = *(_QWORD *)(v9 + 8LL * a2);
    *(_WORD *)(a1 + 192) = 0;
    *(_QWORD *)(a1 + 176) = v10;
    *(_QWORD *)(a1 + 184) = v10 + 48;
  }
  v11 = 0;
  if ( (_DWORD)v7 != 1 )
  {
    v58 = 8 * *(_DWORD *)(a1 + 40);
    v59 = (_QWORD *)sub_BD5C60(*(_QWORD *)a1);
    v11 = (__int64 **)sub_BCCE00(v59, v58);
    if ( !(_DWORD)v7 )
    {
      v92[0] = a1;
LABEL_63:
      sub_2DC34F0(&v94, v92, (__int64 *)&v86);
      v60 = v94;
      v61 = v89;
      v94 = 0;
      v89 = v60;
      v62 = v95;
      v95 = 0;
      v90 = v62;
      v63 = v96;
      v96 = 0;
      v91 = v63;
      if ( v61 )
      {
        j_j___libc_free_0((unsigned __int64)v61);
        if ( v94 )
          j_j___libc_free_0((unsigned __int64)v94);
      }
      while ( v90 - (_QWORD)v89 != 8 )
      {
        sub_2DC34F0(&v94, v92, (__int64 *)&v89);
        v64 = v94;
        v65 = v89;
        v94 = 0;
        v89 = v64;
        v66 = v95;
        v95 = 0;
        v90 = v66;
        v67 = v96;
        v96 = 0;
        v91 = v67;
        if ( v65 )
        {
          j_j___libc_free_0((unsigned __int64)v65);
          if ( v94 )
            j_j___libc_free_0((unsigned __int64)v94);
        }
      }
      v97 = 257;
      v68 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v84 + 8), 0, 0);
      v33 = sub_92B530((unsigned int **)v71, 0x21u, *v89, v68, (__int64)&v94);
      goto LABEL_33;
    }
  }
  v12 = (unsigned int)*a3;
  v80 = 0;
  do
  {
    v22 = *(_QWORD *)(a1 + 272) + 16 * v12;
    v23 = *(_QWORD *)(v22 + 8);
    v24 = 8 * *(_DWORD *)v22;
    v25 = (_QWORD *)sub_BD5C60(*(_QWORD *)a1);
    v26 = (__int64 **)sub_BCCE00(v25, v24);
    v27 = sub_2DC22F0(a1, v26, 0, v11, v23);
    v29 = *(__int64 **)(a1 + 208);
    v93 = 257;
    v15 = (unsigned __int8 *)v27;
    v16 = (__int64)v28;
    v30 = *v29;
    if ( v76 == 1 )
    {
      v31 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(v30 + 56);
      if ( v31 == sub_928890 )
      {
        if ( *v15 > 0x15u || *v28 > 0x15u )
          goto LABEL_48;
        v77 = v28;
        v81 = v15;
        v32 = sub_AAB310(0x21u, v15, v28);
        v15 = v81;
        v16 = (__int64)v77;
        v33 = v32;
      }
      else
      {
        v79 = v28;
        v83 = v15;
        v57 = v31((__int64)v29, 33u, v15, v28);
        v16 = (__int64)v79;
        v15 = v83;
        v33 = v57;
      }
      if ( v33 )
      {
LABEL_31:
        ++*a3;
        goto LABEL_32;
      }
LABEL_48:
      v78 = v16;
      v82 = v15;
      v97 = 257;
      v33 = (__int64)sub_BD2C40(72, unk_3F10FD0);
      if ( v33 )
      {
        v44 = *((_QWORD *)v82 + 1);
        v45 = *(unsigned __int8 *)(v44 + 8);
        if ( (unsigned int)(v45 - 17) > 1 )
        {
          v49 = sub_BCB2A0(*(_QWORD **)v44);
          v51 = v78;
          v50 = (__int64)v82;
        }
        else
        {
          v46 = *(_DWORD *)(v44 + 32);
          v47 = *(_QWORD **)v44;
          BYTE4(v85) = (_BYTE)v45 == 18;
          LODWORD(v85) = v46;
          v48 = (__int64 *)sub_BCB2A0(v47);
          v49 = sub_BCE1B0(v48, v85);
          v50 = (__int64)v82;
          v51 = v78;
        }
        sub_B523C0(v33, v49, 53, 33, v50, v51, (__int64)&v94, 0, 0, 0);
      }
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 216) + 16LL))(
        *(_QWORD *)(a1 + 216),
        v33,
        v92,
        *(_QWORD *)(v71 + 56),
        *(_QWORD *)(v71 + 64));
      v52 = *(_QWORD *)(a1 + 128);
      v53 = v52 + 16LL * *(unsigned int *)(a1 + 136);
      while ( v53 != v52 )
      {
        v54 = *(_QWORD *)(v52 + 8);
        v55 = *(_DWORD *)v52;
        v52 += 16;
        sub_B99FD0(v33, v55, v54);
      }
      goto LABEL_31;
    }
    v13 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(v30 + 16);
    if ( v13 != sub_9202E0 )
    {
      v70 = v28;
      v75 = v15;
      v56 = v13((__int64)v29, 30u, v15, v28);
      v16 = (__int64)v70;
      v15 = v75;
      v17 = v56;
      goto LABEL_13;
    }
    if ( *v15 <= 0x15u && *v28 <= 0x15u )
    {
      v72 = v15;
      v69 = v28;
      if ( (unsigned __int8)sub_AC47B0(30) )
        v14 = sub_AD5570(30, (__int64)v72, v69, 0, 0);
      else
        v14 = sub_AABE40(0x1Eu, v72, v69);
      v15 = v72;
      v16 = (__int64)v69;
      v17 = v14;
LABEL_13:
      if ( v17 )
        goto LABEL_14;
    }
    v97 = 257;
    v17 = sub_B504D0(30, (__int64)v15, v16, (__int64)&v94, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 216) + 16LL))(
      *(_QWORD *)(a1 + 216),
      v17,
      v92,
      *(_QWORD *)(v71 + 56),
      *(_QWORD *)(v71 + 64));
    v35 = 16LL * *(unsigned int *)(a1 + 136);
    v36 = *(_QWORD *)(a1 + 128);
    v73 = v36 + v35;
    while ( v73 != v36 )
    {
      v37 = *(_QWORD *)(v36 + 8);
      v38 = *(_DWORD *)v36;
      v36 += 16;
      sub_B99FD0(v17, v38, v37);
    }
LABEL_14:
    v84 = v17;
    v93 = 257;
    if ( *(__int64 ***)(v17 + 8) == v11 )
    {
      v20 = v17;
      goto LABEL_20;
    }
    v18 = *(_QWORD *)(a1 + 208);
    v19 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v18 + 120LL);
    if ( v19 != sub_920130 )
    {
      v20 = v19(v18, 39u, (_BYTE *)v17, (__int64)v11);
      goto LABEL_19;
    }
    if ( *(_BYTE *)v17 <= 0x15u )
    {
      if ( (unsigned __int8)sub_AC4810(0x27u) )
        v20 = sub_ADAB70(39, v17, v11, 0);
      else
        v20 = sub_AA93C0(0x27u, v17, (__int64)v11);
LABEL_19:
      if ( v20 )
        goto LABEL_20;
    }
    v97 = 257;
    v39 = sub_BD2C40(72, 1u);
    v20 = (__int64)v39;
    if ( v39 )
      sub_B515B0((__int64)v39, v17, (__int64)v11, (__int64)&v94, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 216) + 16LL))(
      *(_QWORD *)(a1 + 216),
      v20,
      v92,
      *(_QWORD *)(v71 + 56),
      *(_QWORD *)(v71 + 64));
    v40 = *(_QWORD *)(a1 + 128);
    v41 = v40;
    v74 = v40 + 16LL * *(unsigned int *)(a1 + 136);
    if ( v40 != v74 )
    {
      do
      {
        v42 = *(_QWORD *)(v41 + 8);
        v43 = *(_DWORD *)v41;
        v41 += 16;
        sub_B99FD0(v20, v43, v42);
      }
      while ( v74 != v41 );
    }
LABEL_20:
    v84 = v20;
    v21 = v87;
    if ( v87 == v88 )
    {
      sub_9281F0((__int64)&v86, v87, &v84);
    }
    else
    {
      if ( v87 )
      {
        *(_QWORD *)v87 = v20;
        v21 = v87;
      }
      v87 = v21 + 8;
    }
    ++v80;
    v12 = (unsigned int)(*a3 + 1);
    *a3 = v12;
  }
  while ( v76 > v80 );
  v33 = 0;
LABEL_32:
  v92[0] = a1;
  if ( !v33 )
    goto LABEL_63;
LABEL_33:
  if ( v89 )
    j_j___libc_free_0((unsigned __int64)v89);
  if ( v86 )
    j_j___libc_free_0(v86);
  return v33;
}
