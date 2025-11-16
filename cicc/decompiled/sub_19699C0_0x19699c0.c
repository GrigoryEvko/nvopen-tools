// Function: sub_19699C0
// Address: 0x19699c0
//
__int64 __fastcall sub_19699C0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        __m128i a7,
        __m128i a8)
{
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rax
  unsigned __int64 v15; // r8
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r14
  _QWORD *v24; // rax
  unsigned int v25; // eax
  __int64 v26; // rsi
  __int64 v27; // rcx
  unsigned __int64 v28; // r9
  int v29; // eax
  __int64 v30; // rdx
  int v31; // eax
  unsigned int v32; // esi
  int v33; // eax
  unsigned int v34; // eax
  __int64 v35; // rsi
  __int64 v36; // r10
  unsigned __int64 v37; // r9
  __int64 v38; // rax
  _QWORD *v39; // rax
  __int64 v40; // rax
  __int64 *v41; // r14
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned int v45; // esi
  int v46; // eax
  __int64 v47; // rax
  _QWORD *v48; // rax
  __int64 v49; // rax
  unsigned int v50; // esi
  int v51; // eax
  __int64 v52; // rax
  _QWORD *v53; // rax
  __int64 v54; // [rsp+8h] [rbp-98h]
  __int64 v55; // [rsp+10h] [rbp-90h]
  unsigned __int64 v56; // [rsp+10h] [rbp-90h]
  unsigned __int64 v57; // [rsp+18h] [rbp-88h]
  unsigned __int64 v58; // [rsp+18h] [rbp-88h]
  __int64 v59; // [rsp+20h] [rbp-80h]
  __int64 v60; // [rsp+20h] [rbp-80h]
  __int64 v61; // [rsp+20h] [rbp-80h]
  __int64 v62; // [rsp+20h] [rbp-80h]
  __int64 v63; // [rsp+28h] [rbp-78h]
  __int64 v64; // [rsp+28h] [rbp-78h]
  __int64 v65; // [rsp+28h] [rbp-78h]
  __int64 v66; // [rsp+28h] [rbp-78h]
  unsigned __int64 v67; // [rsp+28h] [rbp-78h]
  __int64 v68; // [rsp+28h] [rbp-78h]
  unsigned __int64 v69; // [rsp+28h] [rbp-78h]
  __int64 v70; // [rsp+30h] [rbp-70h]
  unsigned __int64 v71; // [rsp+30h] [rbp-70h]
  unsigned __int64 v72; // [rsp+30h] [rbp-70h]
  unsigned __int64 v73; // [rsp+30h] [rbp-70h]
  unsigned __int64 v74; // [rsp+30h] [rbp-70h]
  unsigned __int64 v75; // [rsp+30h] [rbp-70h]
  unsigned __int64 v76; // [rsp+30h] [rbp-70h]
  unsigned __int64 v77; // [rsp+30h] [rbp-70h]
  __int64 v78; // [rsp+38h] [rbp-68h]
  __int64 v79; // [rsp+38h] [rbp-68h]
  __int64 v80; // [rsp+38h] [rbp-68h]
  __int64 v81; // [rsp+38h] [rbp-68h]
  __int64 v82; // [rsp+38h] [rbp-68h]
  __int64 v83; // [rsp+38h] [rbp-68h]
  __int64 v84; // [rsp+38h] [rbp-68h]
  __int64 v85; // [rsp+38h] [rbp-68h]
  __int64 v86; // [rsp+38h] [rbp-68h]
  __int64 v87; // [rsp+38h] [rbp-68h]
  __int64 v88; // [rsp+40h] [rbp-60h]
  __int64 v89; // [rsp+40h] [rbp-60h]
  __int64 v90; // [rsp+40h] [rbp-60h]
  __int64 v91; // [rsp+40h] [rbp-60h]
  __int64 v92; // [rsp+40h] [rbp-60h]
  unsigned __int64 v93; // [rsp+40h] [rbp-60h]
  __int64 v94; // [rsp+40h] [rbp-60h]
  unsigned __int64 v95; // [rsp+40h] [rbp-60h]
  __int64 *v97; // [rsp+50h] [rbp-50h] BYREF
  __int64 v98; // [rsp+58h] [rbp-48h]
  __int64 v99; // [rsp+60h] [rbp-40h] BYREF
  __int64 v100; // [rsp+68h] [rbp-38h]

  v12 = sub_1456040(a1);
  v13 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v12 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v30 = *(_QWORD *)(v12 + 32);
        v12 = *(_QWORD *)(v12 + 24);
        v13 *= v30;
        continue;
      case 1:
        v14 = 16;
        break;
      case 2:
        v14 = 32;
        break;
      case 3:
      case 9:
        v14 = 64;
        break;
      case 4:
        v14 = 80;
        break;
      case 5:
      case 6:
        v14 = 128;
        break;
      case 7:
        v92 = v13;
        v31 = sub_15A9520(a5, 0);
        v13 = v92;
        v14 = (unsigned int)(8 * v31);
        break;
      case 0xB:
        v14 = *(_DWORD *)(v12 + 8) >> 8;
        break;
      case 0xD:
        v89 = v13;
        v24 = (_QWORD *)sub_15A9930(a5, v12);
        v13 = v89;
        v14 = 8LL * *v24;
        break;
      case 0xE:
        v70 = v13;
        v90 = *(_QWORD *)(v12 + 32);
        v78 = *(_QWORD *)(v12 + 24);
        v25 = sub_15A9FE0(a5, v78);
        v26 = v78;
        v13 = v70;
        v27 = 1;
        v28 = v25;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v26 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v44 = *(_QWORD *)(v26 + 32);
              v26 = *(_QWORD *)(v26 + 24);
              v27 *= v44;
              continue;
            case 1:
              v42 = 16;
              goto LABEL_43;
            case 2:
              v42 = 32;
              goto LABEL_43;
            case 3:
            case 9:
              v42 = 64;
              goto LABEL_43;
            case 4:
              v42 = 80;
              goto LABEL_43;
            case 5:
            case 6:
              v42 = 128;
              goto LABEL_43;
            case 7:
              v64 = v70;
              v45 = 0;
              v72 = v28;
              v82 = v27;
              goto LABEL_53;
            case 0xB:
              v42 = *(_DWORD *)(v26 + 8) >> 8;
              goto LABEL_43;
            case 0xD:
              v66 = v70;
              v74 = v28;
              v84 = v27;
              v48 = (_QWORD *)sub_15A9930(a5, v26);
              v27 = v84;
              v28 = v74;
              v13 = v66;
              v42 = 8LL * *v48;
              goto LABEL_43;
            case 0xE:
              v55 = v70;
              v57 = v28;
              v59 = v27;
              v65 = *(_QWORD *)(v26 + 24);
              v83 = *(_QWORD *)(v26 + 32);
              v73 = (unsigned int)sub_15A9FE0(a5, v65);
              v47 = sub_127FA20(a5, v65);
              v27 = v59;
              v28 = v57;
              v13 = v55;
              v42 = 8 * v83 * v73 * ((v73 + ((unsigned __int64)(v47 + 7) >> 3) - 1) / v73);
              goto LABEL_43;
            case 0xF:
              v64 = v70;
              v72 = v28;
              v82 = v27;
              v45 = *(_DWORD *)(v26 + 8) >> 8;
LABEL_53:
              v46 = sub_15A9520(a5, v45);
              v27 = v82;
              v28 = v72;
              v13 = v64;
              v42 = (unsigned int)(8 * v46);
LABEL_43:
              v14 = 8 * v90 * v28 * ((v28 + ((unsigned __int64)(v42 * v27 + 7) >> 3) - 1) / v28);
              break;
          }
          break;
        }
        break;
      case 0xF:
        v91 = v13;
        v29 = sub_15A9520(a5, *(_DWORD *)(v12 + 8) >> 8);
        v13 = v91;
        v14 = (unsigned int)(8 * v29);
        break;
    }
    break;
  }
  v15 = v14 * v13;
  v16 = a2;
  v17 = 1;
  while ( 1 )
  {
    switch ( *(_BYTE *)(v16 + 8) )
    {
      case 1:
        v18 = 16;
        goto LABEL_9;
      case 2:
        v18 = 32;
        goto LABEL_9;
      case 3:
      case 9:
        v18 = 64;
        goto LABEL_9;
      case 4:
        v18 = 80;
        goto LABEL_9;
      case 5:
      case 6:
        v18 = 128;
        goto LABEL_9;
      case 7:
        v79 = v17;
        v32 = 0;
        v93 = v15;
        goto LABEL_29;
      case 0xB:
        v18 = *(_DWORD *)(v16 + 8) >> 8;
        goto LABEL_9;
      case 0xD:
        v81 = v17;
        v95 = v15;
        v39 = (_QWORD *)sub_15A9930(a5, v16);
        v15 = v95;
        v17 = v81;
        v18 = 8LL * *v39;
        goto LABEL_9;
      case 0xE:
        v63 = v17;
        v71 = v15;
        v80 = *(_QWORD *)(v16 + 24);
        v94 = *(_QWORD *)(v16 + 32);
        v34 = sub_15A9FE0(a5, v80);
        v35 = v80;
        v17 = v63;
        v36 = 1;
        v15 = v71;
        v37 = v34;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v35 + 8) )
          {
            case 1:
              v43 = 16;
              goto LABEL_46;
            case 2:
              v43 = 32;
              goto LABEL_46;
            case 3:
            case 9:
              v43 = 64;
              goto LABEL_46;
            case 4:
              v43 = 80;
              goto LABEL_46;
            case 5:
            case 6:
              v43 = 128;
              goto LABEL_46;
            case 7:
              v60 = v63;
              v50 = 0;
              v67 = v71;
              v75 = v37;
              v85 = v36;
              goto LABEL_62;
            case 0xB:
              v43 = *(_DWORD *)(v35 + 8) >> 8;
              goto LABEL_46;
            case 0xD:
              v62 = v63;
              v69 = v71;
              v77 = v37;
              v87 = v36;
              v53 = (_QWORD *)sub_15A9930(a5, v35);
              v36 = v87;
              v37 = v77;
              v15 = v69;
              v17 = v62;
              v43 = 8LL * *v53;
              goto LABEL_46;
            case 0xE:
              v54 = v63;
              v56 = v71;
              v58 = v37;
              v61 = v36;
              v68 = *(_QWORD *)(v35 + 24);
              v86 = *(_QWORD *)(v35 + 32);
              v76 = (unsigned int)sub_15A9FE0(a5, v68);
              v52 = sub_127FA20(a5, v68);
              v36 = v61;
              v37 = v58;
              v15 = v56;
              v17 = v54;
              v43 = 8 * v86 * v76 * ((v76 + ((unsigned __int64)(v52 + 7) >> 3) - 1) / v76);
              goto LABEL_46;
            case 0xF:
              v60 = v63;
              v67 = v71;
              v75 = v37;
              v50 = *(_DWORD *)(v35 + 8) >> 8;
              v85 = v36;
LABEL_62:
              v51 = sub_15A9520(a5, v50);
              v36 = v85;
              v37 = v75;
              v15 = v67;
              v17 = v60;
              v43 = (unsigned int)(8 * v51);
LABEL_46:
              v18 = 8 * v94 * v37 * ((v37 + ((unsigned __int64)(v43 * v36 + 7) >> 3) - 1) / v37);
              goto LABEL_9;
            case 0x10:
              v49 = *(_QWORD *)(v35 + 32);
              v35 = *(_QWORD *)(v35 + 24);
              v36 *= v49;
              continue;
            default:
              goto LABEL_3;
          }
        }
      case 0xF:
        v79 = v17;
        v93 = v15;
        v32 = *(_DWORD *)(v16 + 8) >> 8;
LABEL_29:
        v33 = sub_15A9520(a5, v32);
        v15 = v93;
        v17 = v79;
        v18 = (unsigned int)(8 * v33);
LABEL_9:
        if ( v18 * v17 > v15
          && (v19 = sub_1456040(a1),
              v20 = sub_145CF80((__int64)a6, v19, 1, 0),
              v21 = sub_1480620((__int64)a6, v20, 0),
              (unsigned __int8)sub_148B410((__int64)a6, a4, 0x21u, a1, v21)) )
        {
          v40 = sub_1456040(a1);
          v100 = sub_145CF80((__int64)a6, v40, 1, 0);
          v99 = a1;
          v97 = &v99;
          v98 = 0x200000002LL;
          v41 = sub_147DD40((__int64)a6, (__int64 *)&v97, 2u, 0, a7, a8);
          if ( v97 != &v99 )
            _libc_free((unsigned __int64)v97);
          v22 = sub_14747F0((__int64)a6, (__int64)v41, a2, 0);
        }
        else
        {
          v88 = sub_145CF80((__int64)a6, a2, 1, 0);
          v99 = sub_1483B20(a6, a1, a2, a7, a8);
          v100 = v88;
          v97 = &v99;
          v98 = 0x200000002LL;
          v22 = (__int64)sub_147DD40((__int64)a6, (__int64 *)&v97, 2u, 0, a7, a8);
          if ( v97 != &v99 )
            _libc_free((unsigned __int64)v97);
        }
        if ( a3 != 1 )
        {
          v100 = sub_145CF80((__int64)a6, a2, a3, 0);
          v99 = v22;
          v97 = &v99;
          v98 = 0x200000002LL;
          v22 = sub_147EE30(a6, &v97, 2u, 0, a7, a8);
          if ( v97 != &v99 )
            _libc_free((unsigned __int64)v97);
        }
        return v22;
      case 0x10:
        v38 = *(_QWORD *)(v16 + 32);
        v16 = *(_QWORD *)(v16 + 24);
        v17 *= v38;
        break;
      default:
LABEL_3:
        BUG();
    }
  }
}
