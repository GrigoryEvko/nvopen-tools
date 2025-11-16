// Function: sub_112BDE0
// Address: 0x112bde0
//
unsigned __int8 *__fastcall sub_112BDE0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int8 *result; // rax
  __int16 v4; // dx
  __int64 v7; // rsi
  __int64 v8; // r15
  int v9; // ecx
  char v10; // di
  int v11; // edx
  char v12; // al
  unsigned int *v13; // r12
  __int64 v14; // rdx
  size_t v15; // rdx
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r10
  __int64 v21; // r15
  __int64 v22; // r13
  __int64 v23; // r12
  __int64 v24; // rcx
  __int64 v25; // rdx
  char v26; // al
  __int64 v27; // rax
  char **v28; // rdi
  __int64 v29; // rdx
  bool v30; // al
  __int64 v31; // rdx
  __int64 v32; // r12
  __int64 v33; // rdi
  __int64 *v34; // rax
  __int64 v35; // r14
  __int64 v36; // rdx
  unsigned int v37; // eax
  __int64 v38; // rax
  unsigned int **v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 v44; // r12
  __int64 v45; // r13
  _QWORD *v46; // rax
  __int64 *v47; // r10
  __int64 v48; // r15
  __int64 v49; // r12
  __int64 v50; // rdx
  unsigned int v51; // esi
  __int64 v52; // rax
  __int64 v53; // r12
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r12
  __int64 *v57; // rcx
  __int64 v58; // rdx
  char v59; // al
  unsigned int **v60; // rdi
  __int64 v61; // rax
  __int64 v62; // r8
  unsigned __int8 v63; // al
  _DWORD *v64; // rsi
  unsigned int v65; // edx
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // r12
  __int64 v69; // rdx
  __int64 v70; // rcx
  unsigned int v71; // eax
  __int64 *v72; // r12
  int v73; // ecx
  int v74; // eax
  unsigned int **v75; // rdi
  __int64 v76; // rsi
  __int64 v77; // r13
  __int64 v78; // rax
  __int64 v79; // r12
  __int64 v80; // rax
  __int64 v81; // r12
  char v82; // [rsp+10h] [rbp-E0h]
  char **v83; // [rsp+10h] [rbp-E0h]
  int v84; // [rsp+18h] [rbp-D8h]
  __int64 v85; // [rsp+18h] [rbp-D8h]
  unsigned int v86; // [rsp+18h] [rbp-D8h]
  __int16 v87; // [rsp+20h] [rbp-D0h]
  int v88; // [rsp+2Ch] [rbp-C4h]
  __int64 v89; // [rsp+30h] [rbp-C0h]
  __int64 v90; // [rsp+30h] [rbp-C0h]
  __int64 v91; // [rsp+38h] [rbp-B8h]
  __int64 v92; // [rsp+38h] [rbp-B8h]
  unsigned __int8 *v93; // [rsp+38h] [rbp-B8h]
  __int64 v94; // [rsp+38h] [rbp-B8h]
  __int64 *v95; // [rsp+38h] [rbp-B8h]
  unsigned __int8 *v96; // [rsp+38h] [rbp-B8h]
  unsigned __int8 *v97; // [rsp+38h] [rbp-B8h]
  __int64 v98; // [rsp+38h] [rbp-B8h]
  char v99; // [rsp+4Fh] [rbp-A1h] BYREF
  __int64 v100; // [rsp+50h] [rbp-A0h] BYREF
  char **v101; // [rsp+58h] [rbp-98h] BYREF
  __int64 v102; // [rsp+60h] [rbp-90h] BYREF
  __int16 v103; // [rsp+80h] [rbp-70h]
  __int64 *v104; // [rsp+90h] [rbp-60h] BYREF
  __int64 v105; // [rsp+98h] [rbp-58h]
  __int16 v106; // [rsp+B0h] [rbp-40h]

  v2 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)v2 != 78 )
    return 0;
  v4 = *(_WORD *)(a2 + 2);
  v7 = *(_QWORD *)(a2 - 32);
  v87 = v4 & 0x3F;
  v88 = v4 & 0x3F;
  v8 = *(_QWORD *)(*(_QWORD *)(v2 - 32) + 8LL);
  v89 = *(_QWORD *)(v2 - 32);
  v91 = *(_QWORD *)(v2 + 8);
  v9 = *(unsigned __int8 *)(v91 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
  {
    v10 = 1;
    if ( v9 == 18 )
      goto LABEL_27;
    goto LABEL_6;
  }
  v10 = 0;
  if ( v9 != 18 )
  {
LABEL_6:
    if ( (v9 == 17) != v10 )
      goto LABEL_7;
LABEL_27:
    v82 = v4;
    v84 = sub_BCB060(v8);
    if ( v84 != (unsigned int)sub_BCB060(v91) )
      goto LABEL_7;
    v26 = *(_BYTE *)v89;
    if ( *(_BYTE *)v89 <= 0x1Cu )
    {
LABEL_31:
      LOBYTE(v105) = 0;
      v104 = (__int64 *)&v101;
      if ( !(unsigned __int8)sub_991580((__int64)&v104, v7) || (v27 = *(_QWORD *)(v2 + 16)) == 0 || *(_QWORD *)(v27 + 8) )
      {
LABEL_33:
        v7 = *(_QWORD *)(a2 - 32);
        goto LABEL_7;
      }
      if ( !sub_9893F0(v88, (__int64)v101, &v99) )
        goto LABEL_83;
      v104 = &v100;
      if ( !(unsigned __int8)sub_111E3D0(&v104, (_BYTE *)v89) )
      {
        if ( *(_BYTE *)v89 != 74 || !*(_QWORD *)(v89 - 32) )
          goto LABEL_83;
        v100 = *(_QWORD *)(v89 - 32);
      }
      if ( *(_BYTE *)(*(_QWORD *)(v100 + 8) + 8LL) != 6 )
      {
        v62 = v8;
        if ( *(_BYTE *)(v8 + 8) == 6 )
        {
LABEL_85:
          v85 = v62;
          if ( !(unsigned __int8)sub_B2D610(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL), 30)
            && (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 <= 1 )
          {
            v63 = *(_BYTE *)(v85 + 8);
            if ( v63 <= 3u || v63 == 5 )
            {
              v83 = v101;
              v64 = (_DWORD *)sub_BCAC60(v85, 30, (__int64)v101, (__int64)&v101, v85);
              if ( v64 == sub_C33340() )
                sub_C3C640(&v104, (__int64)v64, v83);
              else
                sub_C3B160((__int64)&v104, v64, (__int64 *)v83);
              v86 = sub_C414D0((__int64)&v104);
              sub_91D830(&v104);
              v65 = v86;
              if ( (v86 & 0x264) != 0 )
              {
                if ( v87 == 33 )
                  v65 = ~(_WORD)v86 & 0x3FF;
                v66 = sub_B37A80(a1[4], v89, v65);
                return sub_F162A0((__int64)a1, a2, v66);
              }
            }
          }
          goto LABEL_33;
        }
        v98 = *(_QWORD *)(v100 + 8);
        v71 = sub_BCB060(v98);
        v72 = (__int64 *)sub_BCD140(*(_QWORD **)(a1[4] + 72), v71);
        v73 = *(unsigned __int8 *)(v98 + 8);
        if ( (unsigned int)(v73 - 17) <= 1 )
        {
          v74 = *(_DWORD *)(v98 + 32);
          BYTE4(v102) = (_BYTE)v73 == 18;
          LODWORD(v102) = v74;
          v72 = (__int64 *)sub_BCE1B0(v72, v102);
        }
        v75 = (unsigned int **)a1[4];
        v106 = 257;
        v76 = v100;
        v77 = sub_A83570(v75, v100, (__int64)v72, (__int64)&v104);
        if ( v99 )
        {
          v78 = sub_AD6530((__int64)v72, v76);
          v106 = 257;
          v79 = v78;
          result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
          if ( !result )
            return result;
          v55 = v79;
          v54 = v77;
LABEL_69:
          v96 = result;
          sub_1113300((__int64)result, 40, v54, v55, (__int64)&v104);
          return v96;
        }
        v80 = sub_AD62B0((__int64)v72);
        v106 = 257;
        v81 = v80;
        result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
        if ( !result )
          return result;
        v70 = v81;
        v69 = v77;
LABEL_97:
        v97 = result;
        sub_1113300((__int64)result, 38, v69, v70, (__int64)&v104);
        return v97;
      }
LABEL_83:
      if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 > 1 )
        v62 = v8;
      else
        v62 = **(_QWORD **)(v8 + 16);
      goto LABEL_85;
    }
    if ( v26 != 73 )
    {
LABEL_30:
      if ( v26 == 72 )
      {
        if ( *(_QWORD *)(v89 - 32) )
        {
          v100 = *(_QWORD *)(v89 - 32);
          if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 <= 1 )
          {
            if ( (unsigned __int8)sub_1112510(v7) )
              goto LABEL_71;
          }
        }
      }
      goto LABEL_31;
    }
    if ( !*(_QWORD *)(v89 - 32) )
      goto LABEL_31;
    v100 = *(_QWORD *)(v89 - 32);
    if ( (v82 & 0x37) == 0x20 )
    {
      if ( (unsigned __int8)sub_1112510(v7) )
        goto LABEL_71;
      if ( v87 == 40 )
      {
        v104 = 0;
        if ( (unsigned __int8)sub_993A50(&v104, v7) )
        {
          v52 = sub_AD64C0(*(_QWORD *)(v100 + 8), 1, 0);
          v106 = 257;
          v53 = v52;
          result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
          if ( !result )
            return result;
          v54 = v100;
          v55 = v53;
          goto LABEL_69;
        }
        goto LABEL_62;
      }
    }
    else if ( (v88 == 33 || v88 == 38) && (unsigned __int8)sub_1112510(v7) )
    {
LABEL_71:
      v56 = sub_AD6530(*(_QWORD *)(v100 + 8), v7);
      v106 = 257;
      result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
      if ( result )
      {
        v25 = v100;
        v24 = v56;
        goto LABEL_24;
      }
      return result;
    }
    if ( v87 == 38 )
    {
      v104 = 0;
      if ( (unsigned __int8)sub_995B10(&v104, v7) )
      {
        v67 = sub_AD62B0(*(_QWORD *)(v100 + 8));
        v106 = 257;
        v68 = v67;
        result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
        if ( !result )
          return result;
        v69 = v100;
        v70 = v68;
        goto LABEL_97;
      }
    }
LABEL_62:
    v26 = *(_BYTE *)v89;
    if ( *(_BYTE *)v89 <= 0x1Cu )
      goto LABEL_31;
    goto LABEL_30;
  }
LABEL_7:
  v104 = (__int64 *)&v101;
  LOBYTE(v105) = 0;
  if ( !(unsigned __int8)sub_991580((__int64)&v104, v7) || *(_BYTE *)(v91 + 8) != 12 )
    return 0;
  v11 = *(unsigned __int8 *)(v8 + 8);
  if ( (unsigned int)(v11 - 17) <= 1 )
    LOBYTE(v11) = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
  if ( (_BYTE)v11 != 12 )
    return 0;
  if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 > 1 )
    goto LABEL_13;
  v28 = v101;
  if ( sub_986760((__int64)v101) )
  {
    v29 = *(_QWORD *)(v2 + 16);
    if ( !v29 )
      goto LABEL_13;
    if ( *(_QWORD *)(v29 + 8) )
    {
      v94 = *(_QWORD *)(v2 + 16);
      v30 = sub_9867B0((__int64)v28);
      v31 = v94;
      if ( v30 )
        goto LABEL_38;
LABEL_13:
      v12 = *(_BYTE *)v89;
LABEL_14:
      if ( v12 != 92 )
        return 0;
      v92 = *(_QWORD *)(v89 - 64);
      if ( !v92 )
        return 0;
      if ( !(unsigned __int8)sub_AC2BE0(*(unsigned __int8 **)(v89 - 32)) )
        return 0;
      v13 = *(unsigned int **)(v89 + 72);
      v14 = 4LL * *(unsigned int *)(v89 + 80);
      if ( v14 )
      {
        v15 = v14 - 4;
        if ( v15 )
        {
          if ( memcmp(v13 + 1, *(const void **)(v89 + 72), v15) )
            return 0;
        }
      }
      v16 = *(_QWORD *)(v8 + 24);
      if ( !sub_C489C0((__int64)v101, *(_DWORD *)(v16 + 8) >> 8) )
        return 0;
      v17 = *v13;
      v18 = sub_BCB2D0(*(_QWORD **)(a1[4] + 72));
      v19 = sub_ACD640(v18, v17, 0);
      v20 = a1[4];
      v103 = 257;
      v21 = v19;
      v90 = v20;
      v22 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v20 + 80) + 96LL))(
              *(_QWORD *)(v20 + 80),
              v92,
              v19);
      if ( !v22 )
      {
        v106 = 257;
        v46 = sub_BD2C40(72, 2u);
        v47 = (__int64 *)v90;
        v22 = (__int64)v46;
        if ( v46 )
        {
          sub_B4DE80((__int64)v46, v92, v21, (__int64)&v104, 0, 0);
          v47 = (__int64 *)v90;
        }
        v95 = v47;
        (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v47[11] + 16LL))(
          v47[11],
          v22,
          &v102,
          v47[7],
          v47[8]);
        v48 = *v95;
        v49 = *v95 + 16LL * *((unsigned int *)v95 + 2);
        if ( *v95 != v49 )
        {
          do
          {
            v50 = *(_QWORD *)(v48 + 8);
            v51 = *(_DWORD *)v48;
            v48 += 16;
            sub_B99FD0(v22, v51, v50);
          }
          while ( v49 != v48 );
        }
      }
      sub_C44740((__int64)&v104, v101, *(_DWORD *)(v16 + 8) >> 8);
      v23 = sub_AD8D80(v16, (__int64)&v104);
      sub_969240((__int64 *)&v104);
      v106 = 257;
      result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
      if ( result )
      {
        v24 = v23;
        v25 = v22;
LABEL_24:
        v93 = result;
        sub_1113300((__int64)result, v88, v25, v24, (__int64)&v104);
        return v93;
      }
      return result;
    }
    v57 = (__int64 *)a1[4];
    v58 = *(_QWORD *)(v89 + 16);
    v59 = 0;
    if ( v58 )
      v59 = *(_QWORD *)(v58 + 8) == 0;
    LOBYTE(v104) = 0;
    v40 = sub_F13D80(a1, v89, v59, v57, &v104, 0);
    if ( v40 )
    {
      v60 = (unsigned int **)a1[4];
      v106 = 257;
      v61 = sub_A83570(v60, v40, v91, (__int64)&v104);
      v43 = v91;
      v44 = v61;
      goto LABEL_45;
    }
    if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 > 1 )
      goto LABEL_13;
    v28 = v101;
  }
  if ( !sub_9867B0((__int64)v28) )
    goto LABEL_13;
  v31 = *(_QWORD *)(v2 + 16);
  if ( !v31 )
    goto LABEL_13;
LABEL_38:
  if ( *(_QWORD *)(v31 + 8) )
    goto LABEL_13;
  v12 = *(_BYTE *)v89;
  if ( *(_BYTE *)v89 <= 0x1Cu )
    return 0;
  if ( v12 != 68 && v12 != 69 )
    goto LABEL_14;
  v32 = *(_QWORD *)(v89 - 32);
  if ( !v32 )
    return 0;
  v33 = *(_QWORD *)(v32 + 8);
  if ( *(_BYTE *)(v33 + 8) != 17 )
    return 0;
  v34 = (__int64 *)sub_BCAE30(v33);
  v35 = a1[4];
  v105 = v36;
  v104 = v34;
  v37 = sub_CA1930(&v104);
  v38 = sub_BCD140(*(_QWORD **)(v35 + 72), v37);
  v39 = (unsigned int **)a1[4];
  v40 = v32;
  v41 = v38;
  v106 = 257;
  v42 = sub_A83570(v39, v32, v38, (__int64)&v104);
  v43 = v41;
  v44 = v42;
LABEL_45:
  v45 = sub_AD6530(v43, v40);
  v106 = 257;
  result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
  if ( result )
  {
    v24 = v45;
    v25 = v44;
    goto LABEL_24;
  }
  return result;
}
