// Function: sub_2624F60
// Address: 0x2624f60
//
__int64 __fastcall sub_2624F60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool v5; // zf
  __int64 v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // r13
  __int64 (__fastcall *v9)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v10; // rax
  _BYTE **v11; // rcx
  __int64 v12; // r14
  __int64 v13; // rdi
  __int64 v14; // rax
  char v15; // bl
  unsigned __int8 *v16; // rax
  unsigned __int8 *v17; // r13
  unsigned int *v18; // r14
  __int64 v19; // rbx
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned __int8 *v24; // rbx
  __int64 (__fastcall *v25)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v26; // r14
  _BYTE *v27; // rax
  _QWORD *v29; // rdi
  __int64 v30; // r9
  __int64 v31; // r11
  unsigned int *v32; // r13
  __int64 v33; // rbx
  __int64 v34; // rdx
  unsigned int v35; // esi
  unsigned int *v36; // rbx
  __int64 v37; // r13
  __int64 v38; // rdx
  unsigned int v39; // esi
  unsigned __int8 *v40; // r14
  __int64 v41; // rbx
  unsigned int v42; // eax
  unsigned int v43; // r13d
  unsigned int v44; // eax
  __int64 v45; // rdi
  __int64 (__fastcall *v46)(__int64, unsigned int, unsigned __int8 *, __int64); // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  int v49; // eax
  char v50; // cl
  int v51; // eax
  __int64 v52; // rax
  __int64 v53; // rdi
  unsigned __int8 *v54; // r10
  __int64 (__fastcall *v55)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v56; // rax
  unsigned __int8 *v57; // r13
  __int64 v58; // rax
  __int64 v59; // rdi
  unsigned __int8 *v60; // r10
  __int64 (__fastcall *v61)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v62; // rax
  unsigned __int8 *v63; // r14
  __int64 v64; // rdi
  __int64 (__fastcall *v65)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v66; // r9
  _BYTE *v67; // rax
  unsigned int *v68; // r14
  __int64 v69; // rdx
  unsigned int v70; // esi
  unsigned int *v71; // r13
  __int64 v72; // rdx
  unsigned int v73; // esi
  unsigned int *v74; // r13
  __int64 v75; // r14
  __int64 v76; // rdx
  unsigned int v77; // esi
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // [rsp+8h] [rbp-C8h]
  __int64 v81; // [rsp+10h] [rbp-C0h]
  unsigned int v83; // [rsp+18h] [rbp-B8h]
  unsigned __int8 *v84; // [rsp+18h] [rbp-B8h]
  unsigned __int8 *v85; // [rsp+18h] [rbp-B8h]
  __int64 v86; // [rsp+18h] [rbp-B8h]
  __int64 v87; // [rsp+18h] [rbp-B8h]
  unsigned __int8 *v88; // [rsp+18h] [rbp-B8h]
  unsigned __int8 *v89; // [rsp+18h] [rbp-B8h]
  unsigned __int8 *v91; // [rsp+20h] [rbp-B0h]
  __int64 v92; // [rsp+20h] [rbp-B0h]
  __int64 v93; // [rsp+20h] [rbp-B0h]
  __int64 v94; // [rsp+20h] [rbp-B0h]
  _BYTE *v95; // [rsp+28h] [rbp-A8h] BYREF
  char v96[8]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v97; // [rsp+38h] [rbp-98h]
  _BYTE v98[32]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v99; // [rsp+60h] [rbp-70h]
  const char *v100; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v101; // [rsp+78h] [rbp-58h]
  unsigned __int64 v102; // [rsp+80h] [rbp-50h]
  unsigned int v103; // [rsp+88h] [rbp-48h]
  __int16 v104; // [rsp+90h] [rbp-40h]

  v5 = *(_DWORD *)a3 == 2;
  v95 = (_BYTE *)a4;
  if ( !v5 )
  {
    v6 = *(_QWORD *)(a3 + 32);
    if ( (_BYTE)qword_4FF31A8 && !a1[2] )
    {
      v29 = (_QWORD *)a1[8];
      v30 = *a1;
      v100 = "bits_use";
      v104 = 259;
      v6 = sub_B30500(v29, 0, 8, (__int64)&v100, v6, v30);
    }
    v7 = *(_QWORD *)(a2 + 80);
    v99 = 257;
    v8 = a1[8];
    v9 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v7 + 64LL);
    if ( v9 == sub_920540 )
    {
      if ( sub_BCEA30(a1[8]) )
        goto LABEL_22;
      if ( *(_BYTE *)v6 > 0x15u )
        goto LABEL_22;
      v10 = sub_2619A30(&v95, (__int64)v96);
      if ( v11 != v10 )
        goto LABEL_22;
      LOBYTE(v104) = 0;
      v12 = sub_AD9FD0(v8, (unsigned __int8 *)v6, (__int64 *)&v95, 1, 0, (__int64)&v100, 0);
      if ( (_BYTE)v104 )
      {
        LOBYTE(v104) = 0;
        if ( v103 > 0x40 && v102 )
          j_j___libc_free_0_0(v102);
        if ( v101 > 0x40 && v100 )
          j_j___libc_free_0_0((unsigned __int64)v100);
      }
    }
    else
    {
      v12 = v9(v7, a1[8], (_BYTE *)v6, &v95, 1, 0);
    }
    if ( v12 )
      goto LABEL_9;
LABEL_22:
    v104 = 257;
    v12 = (__int64)sub_BD2C40(88, 2u);
    if ( !v12 )
      goto LABEL_25;
    v31 = *(_QWORD *)(v6 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17 <= 1 )
    {
LABEL_24:
      sub_B44260(v12, v31, 34, 2u, 0, 0);
      *(_QWORD *)(v12 + 72) = v8;
      *(_QWORD *)(v12 + 80) = sub_B4DC50(v8, (__int64)&v95, 1);
      sub_B4D9A0(v12, v6, (__int64 *)&v95, 1, (__int64)&v100);
LABEL_25:
      sub_B4DDE0(v12, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v12,
        v98,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v32 = *(unsigned int **)a2;
      v33 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v33 )
      {
        do
        {
          v34 = *((_QWORD *)v32 + 1);
          v35 = *v32;
          v32 += 4;
          sub_B99FD0(v12, v35, v34);
        }
        while ( (unsigned int *)v33 != v32 );
      }
LABEL_9:
      v13 = *(_QWORD *)(a2 + 48);
      v99 = 257;
      v80 = a1[8];
      v14 = sub_AA4E30(v13);
      v15 = sub_AE5020(v14, v80);
      v104 = 257;
      v16 = (unsigned __int8 *)sub_BD2C40(80, unk_3F10A14);
      v17 = v16;
      if ( v16 )
        sub_B4D190((__int64)v16, v80, v12, (__int64)&v100, 0, v15, 0, 0);
      (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v17,
        v98,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v18 = *(unsigned int **)a2;
      v19 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v19 )
      {
        do
        {
          v20 = *((_QWORD *)v18 + 1);
          v21 = *v18;
          v18 += 4;
          sub_B99FD0((__int64)v17, v21, v20);
        }
        while ( (unsigned int *)v19 != v18 );
      }
      v99 = 257;
      v22 = sub_AD4C50(*(_QWORD *)(a3 + 40), (__int64 **)a1[8], 0);
      v23 = *(_QWORD *)(a2 + 80);
      v24 = (unsigned __int8 *)v22;
      v25 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v23 + 16LL);
      if ( v25 == sub_9202E0 )
      {
        if ( *v17 > 0x15u || *v24 > 0x15u )
          goto LABEL_28;
        if ( (unsigned __int8)sub_AC47B0(28) )
          v26 = sub_AD5570(28, (__int64)v17, v24, 0, 0);
        else
          v26 = sub_AABE40(0x1Cu, v17, v24);
      }
      else
      {
        v26 = v25(v23, 28u, v17, v24);
      }
      if ( v26 )
      {
LABEL_19:
        v104 = 257;
        v27 = (_BYTE *)sub_ACD640(a1[8], 0, 0);
        return sub_92B530((unsigned int **)a2, 0x21u, v26, v27, (__int64)&v100);
      }
LABEL_28:
      v104 = 257;
      v26 = sub_B504D0(28, (__int64)v17, (__int64)v24, (__int64)&v100, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v26,
        v98,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v36 = *(unsigned int **)a2;
      v37 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v37 )
      {
        do
        {
          v38 = *((_QWORD *)v36 + 1);
          v39 = *v36;
          v36 += 4;
          sub_B99FD0(v26, v39, v38);
        }
        while ( (unsigned int *)v37 != v36 );
      }
      goto LABEL_19;
    }
    v48 = *((_QWORD *)v95 + 1);
    v49 = *(unsigned __int8 *)(v48 + 8);
    if ( v49 == 17 )
    {
      v50 = 0;
    }
    else
    {
      v50 = 1;
      if ( v49 != 18 )
        goto LABEL_24;
    }
    v51 = *(_DWORD *)(v48 + 32);
    BYTE4(v97) = v50;
    LODWORD(v97) = v51;
    v31 = sub_BCE1B0((__int64 *)v31, v97);
    goto LABEL_24;
  }
  v40 = (unsigned __int8 *)a4;
  v41 = *(_QWORD *)(*(_QWORD *)(a3 + 48) + 8LL);
  v91 = *(unsigned __int8 **)(a3 + 48);
  v42 = *(_DWORD *)(v41 + 8);
  v99 = 257;
  v81 = *(_QWORD *)(a4 + 8);
  v83 = v42 >> 8;
  v43 = sub_BCB060(v81);
  v44 = sub_BCB060(v41);
  if ( v43 < v44 )
  {
    v40 = (unsigned __int8 *)sub_A82F30((unsigned int **)a2, (__int64)v40, v41, (__int64)v98, 0);
    goto LABEL_51;
  }
  if ( v41 != v81 && v43 != v44 )
  {
    v45 = *(_QWORD *)(a2 + 80);
    v46 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, __int64))(*(_QWORD *)v45 + 120LL);
    if ( v46 == sub_920130 )
    {
      if ( *v40 > 0x15u )
      {
LABEL_90:
        v104 = 257;
        v40 = (unsigned __int8 *)sub_B51D30(38, (__int64)v40, v41, (__int64)&v100, 0, 0);
        (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
          *(_QWORD *)(a2 + 88),
          v40,
          v98,
          *(_QWORD *)(a2 + 56),
          *(_QWORD *)(a2 + 64));
        sub_94AAF0((unsigned int **)a2, (__int64)v40);
        goto LABEL_51;
      }
      if ( (unsigned __int8)sub_AC4810(0x26u) )
        v47 = sub_ADAB70(38, (unsigned __int64)v40, (__int64 **)v41, 0);
      else
        v47 = sub_AA93C0(0x26u, (unsigned __int64)v40, v41);
    }
    else
    {
      v47 = v46(v45, 38u, v40, v41);
    }
    if ( v47 )
    {
      v40 = (unsigned __int8 *)v47;
      goto LABEL_51;
    }
    goto LABEL_90;
  }
LABEL_51:
  v99 = 257;
  v52 = sub_ACD640(v41, v83 - 1, 0);
  v53 = *(_QWORD *)(a2 + 80);
  v54 = (unsigned __int8 *)v52;
  v55 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v53 + 16LL);
  if ( v55 != sub_9202E0 )
  {
    v89 = v54;
    v79 = v55(v53, 28u, v40, v54);
    v54 = v89;
    v57 = (unsigned __int8 *)v79;
    goto LABEL_57;
  }
  if ( *v40 <= 0x15u && *v54 <= 0x15u )
  {
    v84 = v54;
    if ( (unsigned __int8)sub_AC47B0(28) )
      v56 = sub_AD5570(28, (__int64)v40, v84, 0, 0);
    else
      v56 = sub_AABE40(0x1Cu, v40, v84);
    v54 = v84;
    v57 = (unsigned __int8 *)v56;
LABEL_57:
    if ( v57 )
      goto LABEL_58;
  }
  v104 = 257;
  v57 = (unsigned __int8 *)sub_B504D0(28, (__int64)v40, (__int64)v54, (__int64)&v100, 0, 0);
  (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v57,
    v98,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v68 = *(unsigned int **)a2;
  v86 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v86 )
  {
    do
    {
      v69 = *((_QWORD *)v68 + 1);
      v70 = *v68;
      v68 += 4;
      sub_B99FD0((__int64)v57, v70, v69);
    }
    while ( (unsigned int *)v86 != v68 );
  }
LABEL_58:
  v99 = 257;
  v58 = sub_ACD640(v41, 1, 0);
  v59 = *(_QWORD *)(a2 + 80);
  v60 = (unsigned __int8 *)v58;
  v61 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v59 + 32LL);
  if ( v61 != sub_9201A0 )
  {
    v88 = v60;
    v78 = v61(v59, 25u, v60, v57, 0, 0);
    v60 = v88;
    v63 = (unsigned __int8 *)v78;
    goto LABEL_64;
  }
  if ( *v60 <= 0x15u && *v57 <= 0x15u )
  {
    v85 = v60;
    if ( (unsigned __int8)sub_AC47B0(25) )
      v62 = sub_AD5570(25, (__int64)v85, v57, 0, 0);
    else
      v62 = sub_AABE40(0x19u, v85, v57);
    v60 = v85;
    v63 = (unsigned __int8 *)v62;
LABEL_64:
    if ( v63 )
      goto LABEL_65;
  }
  v104 = 257;
  v63 = (unsigned __int8 *)sub_B504D0(25, (__int64)v60, (__int64)v57, (__int64)&v100, 0, 0);
  (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v63,
    v98,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v71 = *(unsigned int **)a2;
  v87 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v87 )
  {
    do
    {
      v72 = *((_QWORD *)v71 + 1);
      v73 = *v71;
      v71 += 4;
      sub_B99FD0((__int64)v63, v73, v72);
    }
    while ( (unsigned int *)v87 != v71 );
  }
LABEL_65:
  v64 = *(_QWORD *)(a2 + 80);
  v99 = 257;
  v65 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v64 + 16LL);
  if ( v65 == sub_9202E0 )
  {
    if ( *v91 > 0x15u || *v63 > 0x15u )
    {
LABEL_80:
      v104 = 257;
      v93 = sub_B504D0(28, (__int64)v91, (__int64)v63, (__int64)&v100, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v93,
        v98,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v74 = *(unsigned int **)a2;
      v66 = v93;
      v75 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v75 )
      {
        do
        {
          v76 = *((_QWORD *)v74 + 1);
          v77 = *v74;
          v74 += 4;
          v94 = v66;
          sub_B99FD0(v66, v77, v76);
          v66 = v94;
        }
        while ( (unsigned int *)v75 != v74 );
      }
      goto LABEL_71;
    }
    if ( (unsigned __int8)sub_AC47B0(28) )
      v66 = sub_AD5570(28, (__int64)v91, v63, 0, 0);
    else
      v66 = sub_AABE40(0x1Cu, v91, v63);
  }
  else
  {
    v66 = v65(v64, 28u, v91, v63);
  }
  if ( !v66 )
    goto LABEL_80;
LABEL_71:
  v92 = v66;
  v104 = 257;
  v67 = (_BYTE *)sub_ACD640(v41, 0, 0);
  return sub_92B530((unsigned int **)a2, 0x21u, v92, v67, (__int64)&v100);
}
