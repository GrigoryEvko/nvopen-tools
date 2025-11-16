// Function: sub_30DB340
// Address: 0x30db340
//
__int64 __fastcall sub_30DB340(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  unsigned int v4; // r10d
  __int64 v6; // r14
  __int64 *v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  char v10; // r10
  int v11; // edx
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // rax
  int v17; // ebx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rax
  int v22; // edx
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r15
  __int64 v27; // rax
  int v28; // r15d
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r8
  __int64 v32; // rax
  unsigned __int8 *v33; // r15
  __int64 v34; // r9
  __int64 v35; // r12
  unsigned __int8 *v37; // r13
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rcx
  _BYTE *v41; // rbx
  int v42; // ecx
  __int64 v43; // rsi
  int v44; // ecx
  unsigned int v45; // edx
  _QWORD *v46; // rax
  _BYTE *v47; // r10
  __int64 (__fastcall *v48)(_QWORD, __int64); // rax
  __int64 *v49; // rax
  __int64 *v50; // rbx
  __int64 v51; // r10
  __int64 v52; // rax
  _BYTE *v53; // rbx
  __int64 v54; // rax
  _BYTE *v55; // r15
  unsigned __int64 v56; // rcx
  unsigned int v57; // ebx
  unsigned __int64 v58; // rax
  unsigned int v59; // eax
  unsigned int v60; // eax
  bool v61; // zf
  __int64 v62; // rbx
  __int64 *v63; // rax
  __int64 v64; // rax
  __int64 v65; // rcx
  unsigned int v66; // eax
  _BYTE *v67; // rax
  __int64 *v68; // rax
  __int64 v69; // rdi
  __int64 v70; // rsi
  __int64 v71; // rbx
  __int64 v72; // rbx
  __int64 v73; // rax
  unsigned int v74; // ebx
  _BYTE *v75; // rax
  int v76; // eax
  int v77; // edi
  int v78; // edi
  __int64 v79; // r8
  int v80; // edi
  unsigned int v81; // esi
  _QWORD *v82; // rdx
  _BYTE *v83; // r9
  int v84; // eax
  int v85; // edx
  int v86; // r10d
  char v87; // [rsp+8h] [rbp-98h]
  char v88; // [rsp+8h] [rbp-98h]
  __int64 v89; // [rsp+8h] [rbp-98h]
  char v90; // [rsp+8h] [rbp-98h]
  char v91; // [rsp+8h] [rbp-98h]
  char v92; // [rsp+8h] [rbp-98h]
  unsigned __int8 v93; // [rsp+8h] [rbp-98h]
  __int64 v94; // [rsp+10h] [rbp-90h]
  char v95; // [rsp+1Eh] [rbp-82h]
  unsigned __int8 v96; // [rsp+1Fh] [rbp-81h]
  unsigned __int8 v97; // [rsp+28h] [rbp-78h]
  unsigned int v98; // [rsp+28h] [rbp-78h]
  unsigned __int64 v99; // [rsp+28h] [rbp-78h]
  __int64 v100; // [rsp+38h] [rbp-68h] BYREF
  __int64 v101; // [rsp+40h] [rbp-60h] BYREF
  __int64 v102; // [rsp+48h] [rbp-58h]
  _BYTE v103[80]; // [rsp+50h] [rbp-50h] BYREF

  v2 = a1;
  v3 = (__int64)a2;
  v97 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
  if ( !v97 )
    return 1;
  if ( (unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 53) || (unsigned __int8)sub_B49560((__int64)a2, 53) )
  {
    v4 = sub_B2D610(*(_QWORD *)(a1 + 72), 53);
    if ( !(_BYTE)v4 )
    {
      *(_BYTE *)(a1 + 106) = 1;
      return v4;
    }
  }
  if ( *a2 == 85 && ((unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 27) || (unsigned __int8)sub_B49560((__int64)a2, 27)) )
    *(_BYTE *)(a1 + 108) = 1;
  v6 = *((_QWORD *)a2 - 4);
  if ( v6 && !*(_BYTE *)v6 && *(_QWORD *)(v6 + 24) == *((_QWORD *)a2 + 10) )
  {
    v94 = *((_QWORD *)a2 - 4);
    v96 = 0;
  }
  else
  {
    v101 = *((_QWORD *)a2 - 4);
    v7 = sub_30D7570(a1 + 136, &v101);
    if ( !v7 || (v6 = v7[1]) == 0 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *((_QWORD *)a2 + 10) )
    {
      (*(void (__fastcall **)(__int64, unsigned __int8 *))(*(_QWORD *)a1 + 120LL))(a1, a2);
      if ( !(unsigned __int8)sub_B49E20((__int64)a2) && *(_BYTE *)(a1 + 456) )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 80LL))(a1);
        *(_BYTE *)(a1 + 456) = 0;
      }
      return sub_30D2DF0(v2, v3);
    }
    v94 = 0;
    v96 = v97;
  }
  v10 = sub_971E80((__int64)a2, v6);
  if ( !v10 )
    goto LABEL_57;
  v11 = *a2;
  v101 = (__int64)v103;
  v102 = 0x400000000LL;
  if ( v11 == 40 )
  {
    v91 = v10;
    v60 = sub_B491D0((__int64)a2);
    v10 = v91;
    v12 = 32LL * v60;
  }
  else
  {
    v12 = 0;
    if ( v11 != 85 )
    {
      v12 = 64;
      if ( v11 != 34 )
LABEL_156:
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_34;
  v87 = v10;
  v13 = sub_BD2BC0((__int64)a2);
  v10 = v87;
  v15 = v13 + v14;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v15 >> 4) )
      goto LABEL_157;
    goto LABEL_34;
  }
  v16 = sub_BD2BC0((__int64)a2);
  v10 = v87;
  if ( !(unsigned int)((v15 - v16) >> 4) )
  {
LABEL_34:
    v20 = 0;
    goto LABEL_35;
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_157;
  v17 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_156;
  v18 = sub_BD2BC0((__int64)a2);
  v10 = v87;
  v20 = 32LL * (unsigned int)(*(_DWORD *)(v18 + v19 - 4) - v17);
LABEL_35:
  v21 = (32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v12 - v20) >> 5;
  if ( HIDWORD(v102) < (unsigned int)v21 )
  {
    v92 = v10;
    sub_C8D5F0((__int64)&v101, v103, (unsigned int)v21, 8u, v8, v9);
    v10 = v92;
  }
  v22 = *a2;
  if ( v22 == 40 )
  {
    v90 = v10;
    v59 = sub_B491D0((__int64)a2);
    v10 = v90;
    v23 = -32 - 32LL * v59;
  }
  else
  {
    v23 = -32;
    if ( v22 != 85 )
    {
      v23 = -96;
      if ( v22 != 34 )
        goto LABEL_156;
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_45;
  v88 = v10;
  v24 = sub_BD2BC0((__int64)a2);
  v10 = v88;
  v26 = v24 + v25;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v26 >> 4) )
      goto LABEL_157;
  }
  else
  {
    v27 = sub_BD2BC0((__int64)a2);
    v10 = v88;
    if ( (unsigned int)((v26 - v27) >> 4) )
    {
      if ( (a2[7] & 0x80u) != 0 )
      {
        v28 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          goto LABEL_156;
        v29 = sub_BD2BC0((__int64)a2);
        v10 = v88;
        v23 -= 32LL * (unsigned int)(*(_DWORD *)(v29 + v30 - 4) - v28);
        goto LABEL_45;
      }
LABEL_157:
      BUG();
    }
  }
LABEL_45:
  v31 = (__int64)&a2[v23];
  v32 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
  v33 = &a2[-v32];
  if ( &a2[-v32] == &a2[v23] )
  {
    v40 = (unsigned int)v102;
    goto LABEL_95;
  }
  v34 = (__int64)a2;
  v95 = v10;
  v35 = v6;
  v37 = &a2[v23];
  while ( 1 )
  {
    v41 = *(_BYTE **)v33;
    if ( **(_BYTE **)v33 > 0x15u )
      break;
LABEL_47:
    v38 = (unsigned int)v102;
    v39 = (unsigned int)v102 + 1LL;
    if ( v39 > HIDWORD(v102) )
    {
      v89 = v34;
      sub_C8D5F0((__int64)&v101, v103, v39, 8u, v31, v34);
      v38 = (unsigned int)v102;
      v34 = v89;
    }
    v33 += 32;
    *(_QWORD *)(v101 + 8 * v38) = v41;
    v40 = (unsigned int)(v102 + 1);
    LODWORD(v102) = v102 + 1;
    if ( v37 == v33 )
    {
      v10 = v95;
      v2 = a1;
      v6 = v35;
      v3 = v34;
LABEL_95:
      v93 = v10;
      v62 = sub_97A150(v3, v6, (__int64 *)v101, v40, 0, 1);
      if ( !v62 )
        goto LABEL_55;
      v100 = v3;
      v63 = sub_30D9190(v2 + 136, &v100);
      v4 = v93;
      *v63 = v62;
      if ( (_BYTE *)v101 != v103 )
      {
        _libc_free(v101);
        return v93;
      }
      return v4;
    }
  }
  v42 = *(_DWORD *)(a1 + 160);
  v43 = *(_QWORD *)(a1 + 144);
  if ( !v42 )
    goto LABEL_54;
  v44 = v42 - 1;
  v45 = v44 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
  v46 = (_QWORD *)(v43 + 16LL * v45);
  v47 = (_BYTE *)*v46;
  if ( v41 == (_BYTE *)*v46 )
  {
LABEL_53:
    v41 = (_BYTE *)v46[1];
    if ( !v41 )
      goto LABEL_54;
    goto LABEL_47;
  }
  v76 = 1;
  while ( v47 != (_BYTE *)-4096LL )
  {
    v77 = v76 + 1;
    v45 = v44 & (v76 + v45);
    v46 = (_QWORD *)(v43 + 16LL * v45);
    v47 = (_BYTE *)*v46;
    if ( v41 == (_BYTE *)*v46 )
      goto LABEL_53;
    v76 = v77;
  }
LABEL_54:
  v2 = a1;
  v6 = v35;
  v3 = v34;
LABEL_55:
  if ( (_BYTE *)v101 != v103 )
    _libc_free(v101);
LABEL_57:
  if ( *(_BYTE *)v3 != 85
    || (v64 = *(_QWORD *)(v3 - 32)) == 0
    || *(_BYTE *)v64
    || (v65 = *(_QWORD *)(v3 + 80), *(_QWORD *)(v64 + 24) != v65)
    || (*(_BYTE *)(v64 + 33) & 0x20) == 0 )
  {
    if ( v6 == sub_B43CB0(v3) )
    {
      v61 = *(_BYTE *)(v2 + 457) == 0;
      *(_BYTE *)(v2 + 105) = 1;
      if ( v61 )
        return 0;
    }
    v48 = *(__int64 (__fastcall **)(_QWORD, __int64))(v2 + 48);
    if ( !v48 )
      goto LABEL_77;
    v49 = (__int64 *)v48(*(_QWORD *)(v2 + 56), v6);
    v50 = v49;
    if ( !v49
      || !sub_981210(*v49, v6, (unsigned int *)&v101)
      || (v50[((unsigned __int64)(unsigned int)v101 >> 6) + 1] & (1LL << v101)) != 0
      || (((int)*(unsigned __int8 *)(*v50 + ((unsigned int)v101 >> 2)) >> (2 * (v101 & 3))) & 3) == 0
      || (unsigned int)(v101 - 121) > 3 )
    {
      goto LABEL_77;
    }
    v51 = *(_DWORD *)(v3 + 4) & 0x7FFFFFF;
    v52 = 32 * (2 - v51);
    v53 = *(_BYTE **)(v3 + v52);
    if ( *v53 != 17 )
      v53 = sub_30D2200(v2, *(_QWORD *)(v3 + v52));
    v54 = 32 * (3 - v51);
    v55 = *(_BYTE **)(v3 + v54);
    if ( !v55 )
LABEL_145:
      BUG();
    if ( *v55 != 17 )
      v55 = sub_30D2200(v2, *(_QWORD *)(v3 + v54));
    if ( !v53 || !v55 )
      goto LABEL_77;
    v98 = *((_DWORD *)v53 + 8);
    if ( v98 > 0x40 )
    {
      v84 = sub_C444A0((__int64)(v53 + 24));
      v56 = -1;
      if ( v98 - v84 <= 0x40 )
        v56 = **((_QWORD **)v53 + 3);
    }
    else
    {
      v56 = *((_QWORD *)v53 + 3);
    }
    v57 = *((_DWORD *)v55 + 8);
    if ( v57 > 0x40 )
    {
      v99 = v56;
      if ( v57 - (unsigned int)sub_C444A0((__int64)(v55 + 24)) > 0x40 )
        goto LABEL_79;
      v56 = v99;
      v58 = **((_QWORD **)v55 + 3);
    }
    else
    {
      v58 = *((_QWORD *)v55 + 3);
    }
    if ( v58 < v56 )
    {
LABEL_77:
      if ( (unsigned __int8)sub_DF9C30(*(__int64 **)(v2 + 8), (_BYTE *)v6) )
        (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v2 + 136LL))(v2, v6, v3, v96);
    }
LABEL_79:
    if ( !(unsigned __int8)sub_B49E20(v3) && (v94 || !(unsigned __int8)sub_B2DCE0(v6)) )
    {
      if ( *(_BYTE *)(v2 + 456) )
        sub_30D1170((_BYTE *)v2);
    }
    return sub_30D2DF0(v2, v3);
  }
  v66 = *(_DWORD *)(v64 + 36);
  if ( v66 <= 0xF3 )
  {
    if ( v66 > 0xC1 )
    {
      switch ( v66 )
      {
        case 0xC2u:
        case 0xD8u:
          *(_BYTE *)(v2 + 111) = 1;
          return 0;
        case 0xCEu:
          v67 = *(_BYTE **)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
          if ( !v67 )
            goto LABEL_145;
          if ( *v67 <= 0x15u )
          {
            v68 = *(__int64 **)(v65 + 16);
LABEL_117:
            v69 = *v68;
            v70 = 1;
            goto LABEL_118;
          }
          v78 = *(_DWORD *)(v2 + 160);
          v79 = *(_QWORD *)(v2 + 144);
          if ( !v78 )
            goto LABEL_150;
          v80 = v78 - 1;
          v81 = v80 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
          v82 = (_QWORD *)(v79 + 16LL * v81);
          v83 = (_BYTE *)*v82;
          if ( (_BYTE *)*v82 == v67 )
          {
LABEL_141:
            v68 = *(__int64 **)(v65 + 16);
            v69 = *v68;
            if ( v82[1] )
              goto LABEL_117;
            v70 = 0;
          }
          else
          {
            v85 = 1;
            while ( v83 != (_BYTE *)-4096LL )
            {
              v86 = v85 + 1;
              v81 = v80 & (v85 + v81);
              v82 = (_QWORD *)(v79 + 16LL * v81);
              v83 = (_BYTE *)*v82;
              if ( v67 == (_BYTE *)*v82 )
                goto LABEL_141;
              v85 = v86;
            }
LABEL_150:
            v70 = 0;
            v69 = **(_QWORD **)(v65 + 16);
          }
LABEL_118:
          v71 = sub_AD64C0(v69, v70, 0);
          break;
        case 0xD0u:
          goto LABEL_121;
        case 0xD6u:
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 128LL))(v2);
          return 0;
        case 0xEEu:
        case 0xF1u:
        case 0xF3u:
          if ( *(_BYTE *)(v2 + 456) )
            sub_30D1170((_BYTE *)v2);
          return 0;
        default:
          goto LABEL_107;
      }
LABEL_119:
      v101 = v3;
      *sub_30D9190(v2 + 136, &v101) = v71;
      return v97;
    }
    goto LABEL_107;
  }
  switch ( v66 )
  {
    case 0x15Au:
LABEL_121:
      v72 = sub_30D1740(v2, *(_QWORD *)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
      if ( v72 )
      {
        v101 = v3;
        *sub_30DA630(v2 + 168, &v101) = v72;
        return v97;
      }
      return 1;
    case 0x177u:
      *(_BYTE *)(v2 + 112) = 1;
      return 0;
    case 0x11Au:
      v73 = *(_QWORD *)(v3 + 32 * (3LL - (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
      v74 = *(_DWORD *)(v73 + 32);
      if ( v74 <= 0x40 )
      {
        if ( *(_QWORD *)(v73 + 24) != 1 )
          goto LABEL_129;
      }
      else if ( (unsigned int)sub_C444A0(v73 + 24) != v74 - 1 )
      {
LABEL_129:
        v75 = (_BYTE *)sub_D64C80(v3, *(_QWORD *)(v2 + 80), 0, 1);
        v4 = 0;
        v71 = (__int64)v75;
        if ( !v75 || *v75 > 0x15u )
          return v4;
        goto LABEL_119;
      }
      return 0;
  }
LABEL_107:
  if ( !(unsigned __int8)sub_B49E20(v3) && !sub_988C10(v3) && *(_BYTE *)(v2 + 456) )
    sub_30D1170((_BYTE *)v2);
  return sub_30D2DF0(v2, v3);
}
