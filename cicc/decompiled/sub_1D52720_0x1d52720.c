// Function: sub_1D52720
// Address: 0x1d52720
//
__int64 __fastcall sub_1D52720(__int64 a1)
{
  __int64 *v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // r14
  __int64 (__fastcall *v7)(__int64, unsigned __int8); // r15
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // r14
  _QWORD *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int16 v17; // dx
  __int64 (*v18)(); // rax
  char *v19; // rsi
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // r11
  __int64 v23; // rsi
  __int64 v24; // r14
  __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // r8
  unsigned int v32; // esi
  __int64 v33; // r9
  unsigned int v34; // r13d
  unsigned int v35; // ecx
  __int64 *v36; // rdx
  __int64 v37; // rdi
  _QWORD *v38; // r9
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // rax
  __int64 v42; // r14
  __int64 v43; // rsi
  __int64 v44; // r14
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 *v47; // rdi
  __int64 v48; // rax
  __int64 (*v49)(); // rdx
  __int64 (*v50)(); // rax
  int v51; // eax
  __int64 v52; // rbx
  int v53; // eax
  int v54; // r13d
  __int64 v55; // r11
  __int64 v56; // rcx
  int v57; // edx
  __int64 *v58; // rax
  __int64 v59; // r9
  int v60; // eax
  __int64 v61; // r13
  int v62; // ecx
  int v63; // eax
  int v64; // r15d
  __int64 v65; // r11
  __int64 *v66; // rsi
  __int64 v67; // rcx
  int v68; // edi
  __int64 v69; // r9
  __int64 *v70; // r15
  int v71; // edi
  __int64 *v72; // [rsp+8h] [rbp-98h]
  __int64 v73; // [rsp+10h] [rbp-90h]
  __int64 v74; // [rsp+10h] [rbp-90h]
  int v75; // [rsp+10h] [rbp-90h]
  __int64 v76; // [rsp+10h] [rbp-90h]
  int v77; // [rsp+18h] [rbp-88h]
  __int64 v78; // [rsp+18h] [rbp-88h]
  __int64 v79; // [rsp+20h] [rbp-80h]
  __int64 v80; // [rsp+20h] [rbp-80h]
  __int64 *v81; // [rsp+20h] [rbp-80h]
  __int64 v82; // [rsp+28h] [rbp-78h]
  int v83; // [rsp+28h] [rbp-78h]
  __int64 v84; // [rsp+38h] [rbp-68h] BYREF
  __int64 v85; // [rsp+40h] [rbp-60h] BYREF
  int v86; // [rsp+48h] [rbp-58h]
  __int64 v87; // [rsp+50h] [rbp-50h]
  __int64 v88; // [rsp+58h] [rbp-48h]
  __int64 v89; // [rsp+60h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 248);
  v3 = v2[98];
  v4 = sub_15E38F0(*v2);
  v5 = *(_QWORD *)(a1 + 320);
  v6 = *(_QWORD *)(v3 + 40);
  v79 = v4;
  v7 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v5 + 288LL);
  v8 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 32LL));
  v9 = 8 * sub_15A9520(v8, 0);
  if ( v9 == 32 )
  {
    v10 = 5;
  }
  else if ( v9 > 0x20 )
  {
    v10 = 6;
    if ( v9 != 64 )
    {
      v10 = 0;
      if ( v9 == 128 )
        v10 = 7;
      if ( v7 == sub_1D45FB0 )
        goto LABEL_6;
      goto LABEL_31;
    }
  }
  else
  {
    v10 = 3;
    if ( v9 != 8 )
    {
      LOBYTE(v10) = v9 == 16;
      v10 = (unsigned int)(4 * v10);
    }
  }
  if ( v7 == sub_1D45FB0 )
  {
LABEL_6:
    v82 = *(_QWORD *)(v5 + 8 * (v10 & 7) + 120);
    goto LABEL_7;
  }
LABEL_31:
  v82 = v7(v5, v10);
LABEL_7:
  v11 = sub_157ED20(v6);
  v12 = v11;
  if ( *(_BYTE *)(v11 + 16) == 74 )
  {
    v13 = *(_QWORD *)(v11 + 8);
    if ( v13 )
    {
      while ( 1 )
      {
        v14 = sub_1648700(v13);
        if ( *((_BYTE *)v14 + 16) == 78 )
        {
          v15 = *(v14 - 3);
          if ( !*(_BYTE *)(v15 + 16)
            && (*(_BYTE *)(v15 + 33) & 0x20) != 0
            && (unsigned int)(*(_DWORD *)(v15 + 36) - 42) <= 1 )
          {
            break;
          }
        }
        v13 = *(_QWORD *)(v13 + 8);
        if ( !v13 )
          return 1;
      }
      v16 = *(_QWORD *)(a1 + 320);
      v17 = 0;
      v77 = 0;
      v18 = *(__int64 (**)())(*(_QWORD *)v16 + 488LL);
      if ( v18 != sub_1D45FC0 )
      {
        v17 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v18)(v16, v79, 0);
        v77 = v17;
      }
      LOWORD(v85) = v17;
      HIDWORD(v85) = -1;
      v19 = *(char **)(v3 + 160);
      if ( v19 == *(char **)(v3 + 168) )
      {
        sub_1D4B220((char **)(v3 + 152), v19, &v85);
      }
      else
      {
        if ( v19 )
        {
          *(_QWORD *)v19 = v85;
          v19 = *(char **)(v3 + 160);
        }
        *(_QWORD *)(v3 + 160) = v19 + 8;
      }
      v83 = sub_1FE0180(*(_QWORD *)(a1 + 248), v12, v82);
      v21 = **(_QWORD **)(a1 + 280);
      v22 = *(_QWORD *)(*(_QWORD *)(a1 + 312) + 8LL) + 960LL;
      if ( v21 )
      {
        v23 = *(_QWORD *)(v21 + 48);
        v84 = v23;
        if ( v23 )
        {
          v80 = v22;
          sub_1623A60((__int64)&v84, v23, 2);
          v22 = v80;
        }
      }
      else
      {
        v84 = 0;
      }
      v24 = *(_QWORD *)(v3 + 56);
      v81 = *(__int64 **)(*(_QWORD *)(a1 + 248) + 792LL);
      v25 = sub_1E0B640(v24, v22, &v84, 0, v20);
      sub_1DD5BA0(v3 + 16, v25);
      v26 = *v81;
      v27 = *(_QWORD *)v25 & 7LL;
      *(_QWORD *)(v25 + 8) = v81;
      v26 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v25 = v26 | v27;
      *(_QWORD *)(v26 + 8) = v25;
      *v81 = v25 | *v81 & 7;
      v85 = 0x10000000;
      v86 = v83;
      v87 = 0;
      v88 = 0;
      v89 = 0;
      sub_1E1A9C0(v25, v24, &v85);
      v85 = 0x40000000;
      v87 = 0;
      v86 = v77;
      v88 = 0;
      v89 = 0;
      sub_1E1A9C0(v25, v24, &v85);
      if ( v84 )
        sub_161E7C0((__int64)&v84, v84);
    }
    return 1;
  }
  if ( !sub_157F790(v6) )
    return 1;
  v29 = sub_1E0CCF0(*(_QWORD *)(a1 + 256), v3);
  v30 = *(_QWORD *)(a1 + 280);
  v31 = *(_QWORD *)(a1 + 256);
  v78 = v29;
  v32 = *(_DWORD *)(v30 + 752);
  if ( !v32 )
  {
    ++*(_QWORD *)(v30 + 728);
    goto LABEL_49;
  }
  v33 = *(_QWORD *)(v30 + 736);
  v34 = ((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4);
  v35 = (v32 - 1) & v34;
  v36 = (__int64 *)(v33 + 40LL * v35);
  v37 = *v36;
  if ( v3 == *v36 )
  {
    v38 = (_QWORD *)v36[1];
    v39 = *((unsigned int *)v36 + 4);
    goto LABEL_36;
  }
  v75 = 1;
  v58 = 0;
  while ( v37 != -8 )
  {
    if ( v37 != -16 || v58 )
      v36 = v58;
    v35 = (v32 - 1) & (v75 + v35);
    v70 = (__int64 *)(v33 + 40LL * v35);
    v37 = *v70;
    if ( v3 == *v70 )
    {
      v38 = (_QWORD *)v70[1];
      v39 = *((unsigned int *)v70 + 4);
      goto LABEL_36;
    }
    ++v75;
    v58 = v36;
    v36 = (__int64 *)(v33 + 40LL * v35);
  }
  v62 = *(_DWORD *)(v30 + 744);
  if ( !v58 )
    v58 = v36;
  ++*(_QWORD *)(v30 + 728);
  v57 = v62 + 1;
  if ( 4 * (v62 + 1) >= 3 * v32 )
  {
LABEL_49:
    v74 = v31;
    sub_1D52430(v30 + 728, 2 * v32);
    v53 = *(_DWORD *)(v30 + 752);
    if ( v53 )
    {
      v54 = v53 - 1;
      v55 = *(_QWORD *)(v30 + 736);
      v31 = v74;
      LODWORD(v56) = (v53 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v57 = *(_DWORD *)(v30 + 744) + 1;
      v58 = (__int64 *)(v55 + 40LL * (unsigned int)v56);
      v59 = *v58;
      if ( v3 == *v58 )
        goto LABEL_51;
      v71 = 1;
      v66 = 0;
      while ( v59 != -8 )
      {
        if ( !v66 && v59 == -16 )
          v66 = v58;
        v56 = v54 & (unsigned int)(v56 + v71);
        v58 = (__int64 *)(v55 + 40 * v56);
        v59 = *v58;
        if ( v3 == *v58 )
          goto LABEL_51;
        ++v71;
      }
      goto LABEL_67;
    }
LABEL_88:
    ++*(_DWORD *)(v30 + 744);
    BUG();
  }
  if ( v32 - *(_DWORD *)(v30 + 748) - v57 > v32 >> 3 )
    goto LABEL_51;
  v76 = v31;
  sub_1D52430(v30 + 728, v32);
  v63 = *(_DWORD *)(v30 + 752);
  if ( !v63 )
    goto LABEL_88;
  v64 = v63 - 1;
  v65 = *(_QWORD *)(v30 + 736);
  v66 = 0;
  LODWORD(v67) = (v63 - 1) & v34;
  v68 = 1;
  v31 = v76;
  v57 = *(_DWORD *)(v30 + 744) + 1;
  v58 = (__int64 *)(v65 + 40LL * (unsigned int)v67);
  v69 = *v58;
  if ( v3 == *v58 )
    goto LABEL_51;
  while ( v69 != -8 )
  {
    if ( !v66 && v69 == -16 )
      v66 = v58;
    v67 = v64 & (unsigned int)(v67 + v68);
    v58 = (__int64 *)(v65 + 40 * v67);
    v69 = *v58;
    if ( v3 == *v58 )
      goto LABEL_51;
    ++v68;
  }
LABEL_67:
  if ( v66 )
    v58 = v66;
LABEL_51:
  *(_DWORD *)(v30 + 744) = v57;
  if ( *v58 != -8 )
    --*(_DWORD *)(v30 + 748);
  v38 = v58 + 3;
  *v58 = v3;
  v58[2] = 0x400000000LL;
  v39 = 0;
  v58[1] = (__int64)(v58 + 3);
LABEL_36:
  sub_1E0E7D0(v31, v78, v38, v39);
  v41 = **(_QWORD **)(a1 + 280);
  v42 = *(_QWORD *)(*(_QWORD *)(a1 + 312) + 8LL) + 192LL;
  if ( v41 )
  {
    v43 = *(_QWORD *)(v41 + 48);
    v84 = v43;
    if ( v43 )
      sub_1623A60((__int64)&v84, v43, 2);
  }
  else
  {
    v84 = 0;
  }
  v73 = *(_QWORD *)(v3 + 56);
  v72 = *(__int64 **)(*(_QWORD *)(a1 + 248) + 792LL);
  v44 = sub_1E0B640(v73, v42, &v84, 0, v40);
  sub_1DD5BA0(v3 + 16, v44);
  v45 = *v72;
  v46 = *(_QWORD *)v44 & 7LL;
  *(_QWORD *)(v44 + 8) = v72;
  v45 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v44 = v45 | v46;
  *(_QWORD *)(v45 + 8) = v44;
  *v72 = v44 | *v72 & 7;
  LOBYTE(v85) = 15;
  v87 = 0;
  LODWORD(v85) = v85 & 0xFFF000FF;
  v88 = v78;
  v86 = 0;
  LODWORD(v89) = 0;
  sub_1E1A9C0(v44, v73, &v85);
  if ( v84 )
    sub_161E7C0((__int64)&v84, v84);
  v47 = *(__int64 **)(a1 + 320);
  v48 = *v47;
  v49 = *(__int64 (**)())(*v47 + 488);
  if ( v49 != sub_1D45FC0 )
  {
    v60 = ((__int64 (__fastcall *)(__int64 *, __int64))v49)(v47, v79);
    if ( v60 )
    {
      v61 = *(_QWORD *)(a1 + 248);
      *(_DWORD *)(v61 + 932) = sub_1DD9760(v3, (unsigned __int16)v60, v82);
    }
    v47 = *(__int64 **)(a1 + 320);
    v48 = *v47;
  }
  v50 = *(__int64 (**)())(v48 + 496);
  if ( v50 != sub_1D45FD0 )
  {
    v51 = ((__int64 (__fastcall *)(__int64 *, __int64))v50)(v47, v79);
    if ( v51 )
    {
      v52 = *(_QWORD *)(a1 + 248);
      *(_DWORD *)(v52 + 936) = sub_1DD9760(v3, (unsigned __int16)v51, v82);
    }
  }
  return 1;
}
