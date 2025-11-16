// Function: sub_1B73C70
// Address: 0x1b73c70
//
__int64 __fastcall sub_1B73C70(
        __int64 a1,
        unsigned int a2,
        __int64 *a3,
        __int64 *a4,
        _BYTE *a5,
        double a6,
        double a7,
        double a8)
{
  __int64 v8; // r13
  __int64 v11; // rsi
  _QWORD *v12; // r14
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // r15
  unsigned __int64 v16; // r13
  __int64 **v17; // rsi
  int v18; // ebx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r14
  __int64 v23; // rax
  unsigned __int8 *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r14
  unsigned int v29; // eax
  __int64 v30; // rsi
  __int64 v31; // rcx
  unsigned __int64 v32; // r9
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 *v36; // r15
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rsi
  __int64 v40; // rsi
  unsigned __int8 *v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rsi
  __int64 v48; // rdx
  unsigned __int8 *v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rax
  __int64 v56; // rdi
  __int64 *v57; // r14
  __int64 v58; // rax
  __int64 v59; // rsi
  __int64 v60; // rsi
  __int64 v61; // rsi
  unsigned __int8 *v62; // rsi
  __int64 ***v63; // r12
  __int64 **v64; // rax
  unsigned int v65; // r13d
  _QWORD *v66; // rax
  __int64 **v67; // rax
  __int64 v68; // r12
  _QWORD *v69; // rax
  __int64 v70; // rax
  __int64 *v71; // rbx
  _QWORD *v72; // rax
  __int64 v73; // rax
  __int64 ***v74; // r12
  __int64 **v75; // rax
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 *v78; // rbx
  __int64 v79; // rax
  __int64 v80; // rcx
  __int64 v81; // rsi
  __int64 v82; // rsi
  unsigned __int8 *v83; // rsi
  unsigned __int64 v84; // rax
  unsigned int v85; // esi
  int v86; // eax
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // [rsp+0h] [rbp-C0h]
  unsigned __int64 v90; // [rsp+8h] [rbp-B8h]
  __int64 v91; // [rsp+10h] [rbp-B0h]
  __int64 *v92; // [rsp+18h] [rbp-A8h]
  __int64 *v93; // [rsp+18h] [rbp-A8h]
  __int64 v94; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v95; // [rsp+18h] [rbp-A8h]
  __int64 v98; // [rsp+30h] [rbp-90h]
  __int64 v99; // [rsp+30h] [rbp-90h]
  unsigned __int64 v100; // [rsp+30h] [rbp-90h]
  __int64 v101; // [rsp+30h] [rbp-90h]
  __int64 v102; // [rsp+38h] [rbp-88h]
  __int64 v103; // [rsp+48h] [rbp-78h] BYREF
  __int64 *v104[2]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v105; // [rsp+60h] [rbp-60h]
  _BYTE v106[16]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v107; // [rsp+80h] [rbp-40h]

  v8 = 1;
  v11 = (__int64)a3;
  v12 = (_QWORD *)*a3;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v11 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v33 = *(_QWORD *)(v11 + 32);
        v11 = *(_QWORD *)(v11 + 24);
        v8 *= v33;
        continue;
      case 1:
        v13 = 16;
        break;
      case 2:
        v13 = 32;
        break;
      case 3:
      case 9:
        v13 = 64;
        break;
      case 4:
        v13 = 80;
        break;
      case 5:
      case 6:
        v13 = 128;
        break;
      case 7:
        v13 = 8 * (unsigned int)sub_15A9520((__int64)a5, 0);
        break;
      case 0xB:
        v13 = *(_DWORD *)(v11 + 8) >> 8;
        break;
      case 0xD:
        v13 = 8LL * *(_QWORD *)sub_15A9930((__int64)a5, v11);
        break;
      case 0xE:
        v102 = *(_QWORD *)(v11 + 32);
        v99 = *(_QWORD *)(v11 + 24);
        v29 = sub_15A9FE0((__int64)a5, v99);
        v30 = v99;
        v31 = 1;
        v32 = v29;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v30 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v88 = *(_QWORD *)(v30 + 32);
              v30 = *(_QWORD *)(v30 + 24);
              v31 *= v88;
              continue;
            case 1:
              v84 = 16;
              goto LABEL_80;
            case 2:
              v84 = 32;
              goto LABEL_80;
            case 3:
            case 9:
              v84 = 64;
              goto LABEL_80;
            case 4:
              v84 = 80;
              goto LABEL_80;
            case 5:
            case 6:
              v84 = 128;
              goto LABEL_80;
            case 7:
              v94 = v31;
              v85 = 0;
              v100 = v32;
              goto LABEL_84;
            case 0xB:
              v84 = *(_DWORD *)(v30 + 8) >> 8;
              goto LABEL_80;
            case 0xD:
              JUMPOUT(0x1B746BB);
            case 0xE:
              v89 = v31;
              v90 = v32;
              v101 = *(_QWORD *)(v30 + 32);
              v91 = *(_QWORD *)(v30 + 24);
              v95 = (unsigned int)sub_15A9FE0((__int64)a5, v91);
              v87 = sub_127FA20((__int64)a5, v91);
              v32 = v90;
              v31 = v89;
              v84 = 8 * v101 * v95 * ((v95 + ((unsigned __int64)(v87 + 7) >> 3) - 1) / v95);
              goto LABEL_80;
            case 0xF:
              v94 = v31;
              v100 = v32;
              v85 = *(_DWORD *)(v30 + 8) >> 8;
LABEL_84:
              v86 = sub_15A9520((__int64)a5, v85);
              v32 = v100;
              v31 = v94;
              v84 = (unsigned int)(8 * v86);
LABEL_80:
              v13 = 8 * v102 * v32 * ((v32 + ((v84 * v31 + 7) >> 3) - 1) / v32);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v13 = 8 * (unsigned int)sub_15A9520((__int64)a5, *(_DWORD *)(v11 + 8) >> 8);
        break;
    }
    break;
  }
  v14 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v14 + 16) )
    BUG();
  v15 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
  if ( *(_DWORD *)(v14 + 36) == 137 )
  {
    v16 = (unsigned __int64)(v13 * v8) >> 3;
    if ( v16 != 1 )
    {
      v105 = 257;
      v17 = (__int64 **)sub_1644900(v12, 8 * (int)v16);
      if ( v17 == *(__int64 ***)v15 )
      {
        v98 = v15;
      }
      else if ( *(_BYTE *)(v15 + 16) > 0x10u )
      {
        v107 = 257;
        v98 = sub_15FDE70((_QWORD *)v15, (__int64)v17, (__int64)v106, 0);
        v77 = a4[1];
        if ( v77 )
        {
          v78 = (__int64 *)a4[2];
          sub_157E9D0(v77 + 40, v98);
          v79 = *(_QWORD *)(v98 + 24);
          v80 = *v78;
          *(_QWORD *)(v98 + 32) = v78;
          v80 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v98 + 24) = v80 | v79 & 7;
          *(_QWORD *)(v80 + 8) = v98 + 24;
          *v78 = *v78 & 7 | (v98 + 24);
        }
        sub_164B780(v98, (__int64 *)v104);
        v81 = *a4;
        if ( *a4 )
        {
          v103 = *a4;
          sub_1623A60((__int64)&v103, v81, 2);
          v82 = *(_QWORD *)(v98 + 48);
          if ( v82 )
            sub_161E7C0(v98 + 48, v82);
          v83 = (unsigned __int8 *)v103;
          *(_QWORD *)(v98 + 48) = v103;
          if ( v83 )
            sub_1623210((__int64)&v103, v83, v98 + 48);
        }
      }
      else
      {
        v98 = sub_15A45D0((__int64 ***)v15, v17);
      }
      v15 = v98;
      v18 = 1;
      while ( 1 )
      {
        while ( 1 )
        {
          v105 = 257;
          if ( (unsigned int)(2 * v18) > v16 )
            break;
          v23 = sub_15A0680(*(_QWORD *)v15, (unsigned int)(8 * v18), 0);
          v24 = (unsigned __int8 *)v23;
          if ( *(_BYTE *)(v15 + 16) > 0x10u || *(_BYTE *)(v23 + 16) > 0x10u )
          {
            v107 = 257;
            v50 = sub_15FB440(23, (__int64 *)v15, v23, (__int64)v106, 0);
            v51 = a4[1];
            v27 = v50;
            if ( v51 )
            {
              v93 = (__int64 *)a4[2];
              sub_157E9D0(v51 + 40, v50);
              v52 = *v93;
              v53 = *(_QWORD *)(v27 + 24) & 7LL;
              *(_QWORD *)(v27 + 32) = v93;
              v52 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v27 + 24) = v52 | v53;
              *(_QWORD *)(v52 + 8) = v27 + 24;
              *v93 = *v93 & 7 | (v27 + 24);
            }
            sub_164B780(v27, (__int64 *)v104);
            v24 = (unsigned __int8 *)*a4;
            if ( *a4 )
            {
              v103 = *a4;
              sub_1623A60((__int64)&v103, (__int64)v24, 2);
              v54 = *(_QWORD *)(v27 + 48);
              v25 = v27 + 48;
              if ( v54 )
              {
                sub_161E7C0(v27 + 48, v54);
                v25 = v27 + 48;
              }
              v24 = (unsigned __int8 *)v103;
              *(_QWORD *)(v27 + 48) = v103;
              if ( v24 )
                sub_1623210((__int64)&v103, v24, v25);
            }
          }
          else
          {
            v27 = sub_15A2D50((__int64 *)v15, v23, 0, 0, a6, a7, a8);
          }
          v105 = 257;
          if ( *(_BYTE *)(v27 + 16) <= 0x10u )
          {
            if ( sub_1593BB0(v27, (__int64)v24, v25, v26) )
              goto LABEL_27;
            if ( *(_BYTE *)(v15 + 16) <= 0x10u )
            {
              v15 = sub_15A2D10((__int64 *)v15, v27, a6, a7, a8);
              goto LABEL_27;
            }
          }
          v107 = 257;
          v55 = sub_15FB440(27, (__int64 *)v15, v27, (__int64)v106, 0);
          v56 = a4[1];
          v15 = v55;
          if ( v56 )
          {
            v57 = (__int64 *)a4[2];
            sub_157E9D0(v56 + 40, v55);
            v58 = *(_QWORD *)(v15 + 24);
            v59 = *v57;
            *(_QWORD *)(v15 + 32) = v57;
            v59 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v15 + 24) = v59 | v58 & 7;
            *(_QWORD *)(v59 + 8) = v15 + 24;
            *v57 = *v57 & 7 | (v15 + 24);
          }
          sub_164B780(v15, (__int64 *)v104);
          v60 = *a4;
          if ( *a4 )
          {
            v103 = *a4;
            sub_1623A60((__int64)&v103, v60, 2);
            v61 = *(_QWORD *)(v15 + 48);
            if ( v61 )
              sub_161E7C0(v15 + 48, v61);
            v62 = (unsigned __int8 *)v103;
            *(_QWORD *)(v15 + 48) = v103;
            if ( v62 )
            {
              v18 *= 2;
              sub_1623210((__int64)&v103, v62, v15 + 48);
              goto LABEL_28;
            }
          }
LABEL_27:
          v18 *= 2;
LABEL_28:
          if ( v18 == v16 )
            return sub_1B710B0(v15, (__int64)a3, a4, a5);
        }
        v19 = sub_15A0680(*(_QWORD *)v15, 8, 0);
        if ( *(_BYTE *)(v15 + 16) <= 0x10u && *(_BYTE *)(v19 + 16) <= 0x10u )
          break;
        v107 = 257;
        v34 = sub_15FB440(23, (__int64 *)v15, v19, (__int64)v106, 0);
        v35 = a4[1];
        v22 = v34;
        if ( v35 )
        {
          v36 = (__int64 *)a4[2];
          sub_157E9D0(v35 + 40, v34);
          v37 = *(_QWORD *)(v22 + 24);
          v38 = *v36;
          *(_QWORD *)(v22 + 32) = v36;
          v38 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v22 + 24) = v38 | v37 & 7;
          *(_QWORD *)(v38 + 8) = v22 + 24;
          *v36 = *v36 & 7 | (v22 + 24);
        }
        sub_164B780(v22, (__int64 *)v104);
        v39 = *a4;
        if ( !*a4 )
          goto LABEL_14;
        v103 = *a4;
        sub_1623A60((__int64)&v103, v39, 2);
        v40 = *(_QWORD *)(v22 + 48);
        if ( v40 )
          sub_161E7C0(v22 + 48, v40);
        v41 = (unsigned __int8 *)v103;
        *(_QWORD *)(v22 + 48) = v103;
        if ( !v41 )
          goto LABEL_14;
        sub_1623210((__int64)&v103, v41, v22 + 48);
        v105 = 257;
        if ( *(_BYTE *)(v22 + 16) > 0x10u )
        {
LABEL_48:
          v107 = 257;
          v42 = sub_15FB440(27, (__int64 *)v98, v22, (__int64)v106, 0);
          v43 = a4[1];
          v15 = v42;
          if ( v43 )
          {
            v92 = (__int64 *)a4[2];
            sub_157E9D0(v43 + 40, v42);
            v44 = *v92;
            v45 = *(_QWORD *)(v15 + 24) & 7LL;
            *(_QWORD *)(v15 + 32) = v92;
            v44 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v15 + 24) = v44 | v45;
            *(_QWORD *)(v44 + 8) = v15 + 24;
            *v92 = *v92 & 7 | (v15 + 24);
          }
          sub_164B780(v15, (__int64 *)v104);
          v46 = *a4;
          if ( *a4 )
          {
            v103 = *a4;
            sub_1623A60((__int64)&v103, v46, 2);
            v47 = *(_QWORD *)(v15 + 48);
            v48 = v15 + 48;
            if ( v47 )
            {
              sub_161E7C0(v15 + 48, v47);
              v48 = v15 + 48;
            }
            v49 = (unsigned __int8 *)v103;
            *(_QWORD *)(v15 + 48) = v103;
            if ( v49 )
              sub_1623210((__int64)&v103, v49, v48);
          }
          goto LABEL_18;
        }
LABEL_15:
        v15 = v98;
        if ( !sub_1593BB0(v22, 257, v20, v21) )
        {
          if ( *(_BYTE *)(v98 + 16) > 0x10u )
            goto LABEL_48;
          v15 = sub_15A2D10((__int64 *)v98, v22, a6, a7, a8);
        }
LABEL_18:
        if ( ++v18 == v16 )
          return sub_1B710B0(v15, (__int64)a3, a4, a5);
      }
      v22 = sub_15A2D50((__int64 *)v15, v19, 0, 0, a6, a7, a8);
LABEL_14:
      v105 = 257;
      if ( *(_BYTE *)(v22 + 16) > 0x10u )
        goto LABEL_48;
      goto LABEL_15;
    }
    return sub_1B710B0(v15, (__int64)a3, a4, a5);
  }
  else
  {
    v63 = (__int64 ***)sub_1649C60(v15);
    v64 = *v63;
    if ( *((_BYTE *)*v63 + 8) == 16 )
      v64 = (__int64 **)*v64[2];
    v65 = *((_DWORD *)v64 + 2);
    v66 = (_QWORD *)sub_16498A0((__int64)v63);
    v65 >>= 8;
    v67 = (__int64 **)sub_16471D0(v66, v65);
    v68 = sub_15A4510(v63, v67, 0);
    v69 = (_QWORD *)sub_16498A0(v68);
    v70 = sub_1643360(v69);
    v71 = (__int64 *)sub_159C470(v70, a2, 0);
    v72 = (_QWORD *)sub_16498A0(v68);
    v73 = sub_1643330(v72);
    v104[0] = v71;
    v106[4] = 0;
    v74 = (__int64 ***)sub_15A2E80(v73, v68, v104, 1u, 0, (__int64)v106, 0);
    v75 = (__int64 **)sub_1646BA0(a3, v65);
    v76 = sub_15A4510(v74, v75, 0);
    return sub_14D8290(v76, (__int64)a3, a5);
  }
}
