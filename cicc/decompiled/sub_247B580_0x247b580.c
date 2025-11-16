// Function: sub_247B580
// Address: 0x247b580
//
void __fastcall sub_247B580(__int64 *a1, unsigned __int8 *a2, unsigned int a3, int a4)
{
  unsigned int v5; // r13d
  int v6; // edx
  int v7; // eax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rax
  int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  unsigned __int8 *v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r12
  int v22; // eax
  unsigned __int64 v23; // rbx
  __int64 v24; // r11
  __int64 (__fastcall *v25)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  int v30; // edx
  int v31; // r12d
  __int64 v32; // r12
  unsigned int *v33; // r12
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // r9
  int v39; // edx
  unsigned int v40; // r13d
  int v41; // eax
  __int64 v42; // r12
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rbx
  __int64 v46; // rax
  int v47; // ebx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r12
  int v53; // edx
  __int64 v54; // rsi
  unsigned __int64 v55; // r12
  int v56; // eax
  int v57; // edx
  unsigned int v58; // r13d
  int v59; // eax
  __int64 v60; // rbx
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // rdx
  __int64 v67; // rbx
  unsigned __int8 *v68; // rcx
  __int64 v69; // rax
  unsigned __int64 v70; // rax
  _BYTE *v71; // rax
  __int64 v72; // rax
  int v73; // edx
  __int64 v74; // rsi
  __int64 **v75; // rax
  unsigned __int64 v76; // rax
  __int64 v77; // rsi
  __int64 **v80; // [rsp+28h] [rbp-178h]
  unsigned int *v81; // [rsp+28h] [rbp-178h]
  __int64 v82; // [rsp+28h] [rbp-178h]
  __int64 v83; // [rsp+28h] [rbp-178h]
  int v84; // [rsp+28h] [rbp-178h]
  unsigned int v85[8]; // [rsp+30h] [rbp-170h] BYREF
  __int16 v86; // [rsp+50h] [rbp-150h]
  _QWORD v87[4]; // [rsp+60h] [rbp-140h] BYREF
  __int16 v88; // [rsp+80h] [rbp-120h]
  _BYTE *v89; // [rsp+90h] [rbp-110h] BYREF
  __int64 v90; // [rsp+98h] [rbp-108h]
  _BYTE v91[64]; // [rsp+A0h] [rbp-100h] BYREF
  unsigned int *v92; // [rsp+E0h] [rbp-C0h] BYREF
  unsigned int v93; // [rsp+E8h] [rbp-B8h]
  __int64 v94; // [rsp+118h] [rbp-88h]
  __int64 v95; // [rsp+120h] [rbp-80h]
  __int64 v96; // [rsp+130h] [rbp-70h]
  __int64 v97; // [rsp+138h] [rbp-68h]
  __int64 v98; // [rsp+140h] [rbp-60h]
  int v99; // [rsp+148h] [rbp-58h]

  v5 = 0;
  sub_23D0AB0((__int64)&v92, (__int64)a2, 0, 0, 0);
  v6 = *a2;
  v89 = v91;
  v90 = 0x800000000LL;
  v7 = v6 - 29;
  if ( v6 == 40 )
    goto LABEL_24;
LABEL_2:
  v8 = 0;
  if ( v7 != 56 )
  {
    if ( v7 != 5 )
      goto LABEL_88;
    v8 = 64;
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
LABEL_6:
    v9 = sub_BD2BC0((__int64)a2);
    v11 = v9 + v10;
    v12 = v9 + v10;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v12 >> 4) )
        goto LABEL_25;
    }
    else
    {
      if ( !(unsigned int)((v11 - sub_BD2BC0((__int64)a2)) >> 4) )
        goto LABEL_25;
      if ( (a2[7] & 0x80u) != 0 )
      {
        v13 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v14 = sub_BD2BC0((__int64)a2);
        v16 = 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
        goto LABEL_11;
      }
    }
    BUG();
  }
LABEL_25:
  while ( 1 )
  {
    v16 = 0;
LABEL_11:
    v17 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
    if ( (unsigned int)((v17 - 32 - v8 - v16) >> 5) - a4 <= v5 )
      break;
    if ( (a2[7] & 0x40) != 0 )
      v18 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    else
      v18 = &a2[-v17];
    v21 = sub_246F3F0((__int64)a1, *(_QWORD *)&v18[32 * v5]);
    v22 = *((_DWORD *)a2 + 1);
    v86 = 257;
    v23 = v5 - (unsigned __int64)(v22 & 0x7FFFFFF);
    v24 = *(_QWORD *)(*(_QWORD *)&a2[32 * v23] + 8LL);
    if ( v24 == *(_QWORD *)(v21 + 8) )
    {
      v27 = v21;
      goto LABEL_21;
    }
    v25 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v96 + 120LL);
    if ( v25 != sub_920130 )
    {
      v82 = *(_QWORD *)(*(_QWORD *)&a2[32 * v23] + 8LL);
      v36 = v25(v96, 49u, (_BYTE *)v21, v24);
      v24 = v82;
      v27 = v36;
      goto LABEL_20;
    }
    if ( *(_BYTE *)v21 <= 0x15u )
    {
      v80 = *(__int64 ***)(*(_QWORD *)&a2[32 * v23] + 8LL);
      if ( (unsigned __int8)sub_AC4810(0x31u) )
        v26 = sub_ADAB70(49, v21, v80, 0);
      else
        v26 = sub_AA93C0(0x31u, v21, (__int64)v80);
      v24 = (__int64)v80;
      v27 = v26;
LABEL_20:
      if ( v27 )
        goto LABEL_21;
    }
    v88 = 257;
    v27 = sub_B51D30(49, v21, v24, (__int64)v87, 0, 0);
    if ( (unsigned __int8)sub_920620(v27) )
    {
      v31 = v99;
      if ( v98 )
        sub_B99FD0(v27, 3u, v98);
      sub_B45150(v27, v31);
    }
    (*(void (__fastcall **)(__int64, __int64, unsigned int *, __int64, __int64))(*(_QWORD *)v97 + 16LL))(
      v97,
      v27,
      v85,
      v94,
      v95);
    v32 = 4LL * v93;
    v81 = &v92[v32];
    if ( v92 != &v92[v32] )
    {
      v33 = v92;
      do
      {
        v34 = *((_QWORD *)v33 + 1);
        v35 = *v33;
        v33 += 4;
        sub_B99FD0(v27, v35, v34);
      }
      while ( v81 != v33 );
    }
LABEL_21:
    v28 = (unsigned int)v90;
    v29 = (unsigned int)v90 + 1LL;
    if ( v29 > HIDWORD(v90) )
    {
      sub_C8D5F0((__int64)&v89, v91, v29, 8u, v19, v20);
      v28 = (unsigned int)v90;
    }
    ++v5;
    *(_QWORD *)&v89[8 * v28] = v27;
    v30 = *a2;
    LODWORD(v90) = v90 + 1;
    v7 = v30 - 29;
    if ( v30 != 40 )
      goto LABEL_2;
LABEL_24:
    v8 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
    if ( (a2[7] & 0x80u) != 0 )
      goto LABEL_6;
  }
  v37 = sub_A17190(a2);
  v39 = *a2;
  v40 = v37 - a4;
  v41 = v39 - 29;
  if ( v39 == 40 )
    goto LABEL_54;
LABEL_41:
  v42 = 0;
  if ( v41 != 56 )
  {
    if ( v41 == 5 )
    {
      v42 = 64;
      goto LABEL_44;
    }
LABEL_88:
    BUG();
  }
LABEL_44:
  if ( (a2[7] & 0x80u) != 0 )
  {
LABEL_45:
    v43 = sub_BD2BC0((__int64)a2);
    v45 = v43 + v44;
    v46 = v43 + v44;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v46 >> 4) )
        goto LABEL_55;
    }
    else
    {
      if ( !(unsigned int)((v45 - sub_BD2BC0((__int64)a2)) >> 4) )
        goto LABEL_55;
      if ( (a2[7] & 0x80u) != 0 )
      {
        v47 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v48 = sub_BD2BC0((__int64)a2);
        v50 = 32LL * (unsigned int)(*(_DWORD *)(v48 + v49 - 4) - v47);
        goto LABEL_50;
      }
    }
    BUG();
  }
LABEL_55:
  while ( 1 )
  {
    v50 = 0;
LABEL_50:
    v51 = (unsigned int)v90;
    if ( v40 >= (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v42 - v50) >> 5) )
      break;
    v52 = *(_QWORD *)&a2[32 * (v40 - (unsigned __int64)(*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
    if ( (unsigned __int64)(unsigned int)v90 + 1 > HIDWORD(v90) )
    {
      sub_C8D5F0((__int64)&v89, v91, (unsigned int)v90 + 1LL, 8u, (unsigned int)v90, v38);
      v51 = (unsigned int)v90;
    }
    ++v40;
    *(_QWORD *)&v89[8 * v51] = v52;
    v53 = *a2;
    LODWORD(v90) = v90 + 1;
    v41 = v53 - 29;
    if ( v53 != 40 )
      goto LABEL_41;
LABEL_54:
    v42 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
    if ( (a2[7] & 0x80u) != 0 )
      goto LABEL_45;
  }
  v54 = *((_QWORD *)a2 + 1);
  v88 = 257;
  v85[1] = 0;
  v55 = sub_B35180((__int64)&v92, v54, a3, (__int64)v89, (unsigned int)v90, v85[0], (__int64)v87);
  v56 = sub_A17190(a2);
  v57 = *a2;
  v58 = v56 - a4;
  v59 = v57 - 29;
  if ( v57 == 40 )
    goto LABEL_72;
LABEL_59:
  v60 = 0;
  if ( v59 != 56 )
  {
    if ( v59 != 5 )
      goto LABEL_88;
    v60 = 64;
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
LABEL_63:
    v61 = sub_BD2BC0((__int64)a2);
    v83 = v62 + v61;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v83 >> 4) )
        goto LABEL_73;
    }
    else
    {
      if ( !(unsigned int)((v83 - sub_BD2BC0((__int64)a2)) >> 4) )
        goto LABEL_73;
      if ( (a2[7] & 0x80u) != 0 )
      {
        v84 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v63 = sub_BD2BC0((__int64)a2);
        v65 = 32LL * (unsigned int)(*(_DWORD *)(v63 + v64 - 4) - v84);
        goto LABEL_68;
      }
    }
    BUG();
  }
LABEL_73:
  while ( 1 )
  {
    v65 = 0;
LABEL_68:
    v66 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
    if ( v58 >= (unsigned int)((v66 - 32 - v60 - v65) >> 5) )
      break;
    v67 = *(_QWORD *)(v55 + 8);
    if ( (a2[7] & 0x40) != 0 )
      v68 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    else
      v68 = &a2[-v66];
    v69 = v58++;
    v70 = sub_246F3F0((__int64)a1, *(_QWORD *)&v68[32 * v69]);
    v71 = (_BYTE *)sub_2464970(a1, &v92, v70, v67, 0);
    v87[0] = "_msprop";
    v88 = 259;
    v72 = sub_A82480(&v92, v71, (_BYTE *)v55, (__int64)v87);
    v73 = *a2;
    v55 = v72;
    v59 = v73 - 29;
    if ( v73 != 40 )
      goto LABEL_59;
LABEL_72:
    v60 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
    if ( (a2[7] & 0x80u) != 0 )
      goto LABEL_63;
  }
  v74 = *((_QWORD *)a2 + 1);
  v88 = 257;
  v75 = (__int64 **)sub_2463540(a1, v74);
  v76 = sub_24633A0((__int64 *)&v92, 0x31u, v55, v75, (__int64)v87, 0, v85[0], 0);
  sub_246EF60((__int64)a1, (__int64)a2, v76);
  v77 = *(unsigned int *)(a1[1] + 4);
  if ( (_DWORD)v77 )
  {
    v77 = (__int64)a2;
    sub_2477350((__int64)a1, (__int64)a2);
  }
  if ( v89 != v91 )
    _libc_free((unsigned __int64)v89);
  sub_F94A20(&v92, v77);
}
