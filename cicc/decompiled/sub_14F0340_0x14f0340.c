// Function: sub_14F0340
// Address: 0x14f0340
//
__int64 *__fastcall sub_14F0340(__int64 *a1, _QWORD *a2)
{
  _QWORD *v2; // r15
  __int64 *v3; // r12
  __int64 v5; // r13
  const char *v6; // r14
  _BYTE *v7; // rbx
  __int64 v8; // rax
  char *v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  int v23; // r13d
  unsigned int i; // ebx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  int v34; // r13d
  unsigned int v35; // ebx
  unsigned int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // r15
  unsigned __int64 v41; // r13
  _BYTE *v42; // r14
  const char *v43; // rbx
  __int64 v44; // r12
  __int64 v45; // rdx
  unsigned __int64 v46; // r9
  __int64 v47; // r8
  int v48; // r13d
  unsigned int v49; // ebx
  unsigned int v50; // eax
  __int64 v51; // rax
  __int64 v52; // rdx
  int v53; // r13d
  unsigned int v54; // r12d
  __int64 v55; // rax
  __int64 v56; // rbx
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 v61; // rdx
  __int64 v62; // rax
  unsigned __int64 v63; // rdi
  char *v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rsi
  __int64 v69; // rdx
  unsigned __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // [rsp+0h] [rbp-340h]
  _BYTE *v75; // [rsp+8h] [rbp-338h]
  __int64 v76; // [rsp+10h] [rbp-330h]
  __int64 v77; // [rsp+10h] [rbp-330h]
  unsigned __int64 v78; // [rsp+10h] [rbp-330h]
  __int64 v79; // [rsp+10h] [rbp-330h]
  _BYTE *v80; // [rsp+18h] [rbp-328h]
  _BYTE *v81; // [rsp+18h] [rbp-328h]
  const char *v82; // [rsp+18h] [rbp-328h]
  __int64 v83; // [rsp+18h] [rbp-328h]
  _BYTE *v84; // [rsp+18h] [rbp-328h]
  unsigned __int64 v85; // [rsp+18h] [rbp-328h]
  __int64 v86; // [rsp+20h] [rbp-320h]
  __int64 v87; // [rsp+20h] [rbp-320h]
  __int64 v88; // [rsp+20h] [rbp-320h]
  _QWORD *v89; // [rsp+20h] [rbp-320h]
  __int64 v90; // [rsp+20h] [rbp-320h]
  __int64 *v91; // [rsp+20h] [rbp-320h]
  __int64 v92; // [rsp+20h] [rbp-320h]
  __int64 v93; // [rsp+28h] [rbp-318h]
  unsigned int v94; // [rsp+28h] [rbp-318h]
  __int64 v95; // [rsp+28h] [rbp-318h]
  __int64 *v96; // [rsp+28h] [rbp-318h]
  __int64 v97; // [rsp+28h] [rbp-318h]
  __int64 v98; // [rsp+28h] [rbp-318h]
  __int64 v99; // [rsp+28h] [rbp-318h]
  unsigned int v100; // [rsp+3Ch] [rbp-304h]
  char *v101; // [rsp+40h] [rbp-300h] BYREF
  char v102; // [rsp+50h] [rbp-2F0h]
  char v103; // [rsp+51h] [rbp-2EFh]
  _BYTE *v104; // [rsp+60h] [rbp-2E0h] BYREF
  __int64 v105; // [rsp+68h] [rbp-2D8h]
  _BYTE v106[64]; // [rsp+70h] [rbp-2D0h] BYREF
  char *v107; // [rsp+B0h] [rbp-290h] BYREF
  __int64 v108; // [rsp+B8h] [rbp-288h]
  char v109; // [rsp+C0h] [rbp-280h] BYREF
  char v110; // [rsp+C1h] [rbp-27Fh]
  const char *v111; // [rsp+100h] [rbp-240h] BYREF
  __int64 v112; // [rsp+108h] [rbp-238h]
  _BYTE v113[560]; // [rsp+110h] [rbp-230h] BYREF

  v2 = a2;
  v3 = a1;
  if ( a2[67] != a2[66] )
  {
    v113[1] = 1;
    v111 = "Invalid multiple blocks";
    v113[0] = 3;
    sub_14EE4B0(a1, (__int64)(a2 + 1), (__int64)&v111);
    return v3;
  }
  v5 = (__int64)(a2 + 4);
  v6 = v113;
  v100 = 0;
  v112 = 0x4000000000LL;
  v7 = v106;
  v105 = 0x4000000000LL;
  v111 = v113;
  v104 = v106;
  while ( 2 )
  {
    v8 = sub_14ED070(v5, 0);
    if ( (_DWORD)v8 == 1 )
    {
      if ( v100 == (__int64)(v2[67] - v2[66]) >> 3 )
      {
        *v3 = 1;
        goto LABEL_11;
      }
LABEL_9:
      v110 = 1;
      v9 = "Malformed block";
LABEL_10:
      v107 = v9;
      v109 = 3;
      sub_14EE4B0(v3, (__int64)(v2 + 1), (__int64)&v107);
      goto LABEL_11;
    }
    if ( (v8 & 0xFFFFFFFD) == 0 )
      goto LABEL_9;
    LODWORD(v112) = 0;
    switch ( (unsigned int)sub_1510D70(v5, HIDWORD(v8), &v111, 0) )
    {
      case 1u:
        if ( !(_DWORD)v112 )
          goto LABEL_21;
        v68 = v2[66];
        v69 = *(_QWORD *)v111;
        v70 = (v2[67] - v68) >> 3;
        if ( *(_QWORD *)v111 > v70 )
        {
          sub_14F0190((__int64)(v2 + 66), v69 - v70);
        }
        else if ( *(_QWORD *)v111 < v70 )
        {
          v71 = v68 + 8 * v69;
          if ( v2[67] != v71 )
            v2[67] = v71;
        }
        continue;
      case 2u:
        v15 = sub_1643270(v2[54]);
        v11 = v100;
        v12 = v15;
        goto LABEL_17;
      case 3u:
        v18 = sub_16432A0(v2[54]);
        v11 = v100;
        v12 = v18;
        goto LABEL_17;
      case 4u:
        v16 = sub_16432B0(v2[54]);
        v11 = v100;
        v12 = v16;
        goto LABEL_17;
      case 5u:
        v17 = sub_1643280(v2[54]);
        v11 = v100;
        v12 = v17;
        goto LABEL_17;
      case 6u:
        if ( (_DWORD)v112 != 1 )
          goto LABEL_21;
        v19 = v2[66];
        if ( v100 >= (unsigned __int64)((v2[67] - v19) >> 3) )
          goto LABEL_116;
        if ( *(_QWORD *)(v19 + 8LL * v100) )
        {
          v93 = *(_QWORD *)(v19 + 8LL * v100);
          sub_1643660(v93, v104, (unsigned int)v105);
          v11 = v100;
          v12 = v93;
          *(_QWORD *)(v2[66] + 8LL * v100) = 0;
        }
        else
        {
          v73 = sub_14F0050((__int64)v2, v2[54], (__int64)v104, (unsigned int)v105);
          v11 = v100;
          v12 = v73;
        }
        LODWORD(v105) = 0;
        goto LABEL_17;
      case 7u:
        if ( !(_DWORD)v112 )
          goto LABEL_21;
        if ( (unsigned __int64)(*(_QWORD *)v111 - 1LL) > 0xFFFFFE )
        {
          v110 = 1;
          v9 = "Bitwidth for integer type out of range";
          goto LABEL_10;
        }
        v20 = sub_1644900(v2[54], *(_QWORD *)v111);
        v11 = v100;
        v12 = v20;
        goto LABEL_17;
      case 8u:
        if ( !(_DWORD)v112 )
          goto LABEL_21;
        v94 = 0;
        if ( (unsigned int)v112 == 2 )
          v94 = *((_DWORD *)v111 + 2);
        v21 = sub_14EFEB0(v2, *(_QWORD *)v111);
        if ( !v21 )
          goto LABEL_123;
        v86 = v21;
        if ( !(unsigned __int8)sub_1643F60(v21) )
          goto LABEL_123;
        v22 = sub_1646BA0(v86, v94);
        v11 = v100;
        v12 = v22;
        goto LABEL_17;
      case 9u:
        if ( (unsigned int)v112 <= 2 )
          goto LABEL_21;
        v107 = &v109;
        v108 = 0x800000000LL;
        if ( (_DWORD)v112 != 3 )
        {
          v87 = v5;
          v23 = v112;
          v80 = v7;
          for ( i = 3; i != v23; ++i )
          {
            v25 = sub_14EFEB0(v2, *(_QWORD *)&v111[8 * i]);
            if ( !v25 )
              break;
            v26 = (unsigned int)v108;
            if ( (unsigned int)v108 >= HIDWORD(v108) )
            {
              v76 = v25;
              sub_16CD150(&v107, &v109, 0, 8);
              v26 = (unsigned int)v108;
              v25 = v76;
            }
            *(_QWORD *)&v107[8 * v26] = v25;
            LODWORD(v108) = v108 + 1;
          }
          v5 = v87;
          v7 = v80;
        }
        v60 = sub_14EFEB0(v2, *((_QWORD *)v111 + 2));
        if ( v60 )
        {
          v61 = (unsigned int)v108;
          if ( (unsigned __int64)(unsigned int)v112 - 3 <= (unsigned int)v108 )
            goto LABEL_98;
        }
        goto LABEL_103;
      case 0xAu:
        v27 = sub_1643290(v2[54]);
        v11 = v100;
        v12 = v27;
        goto LABEL_17;
      case 0xBu:
        if ( (unsigned int)v112 <= 1 )
          goto LABEL_21;
        v58 = sub_14EFEB0(v2, *((_QWORD *)v111 + 1));
        if ( !v58 )
          goto LABEL_123;
        v98 = v58;
        if ( !(unsigned __int8)sub_1643EC0(v58) )
          goto LABEL_123;
        v59 = sub_1645D80(v98, *(_QWORD *)v111);
        v11 = v100;
        v12 = v59;
        goto LABEL_17;
      case 0xCu:
        if ( (unsigned int)v112 <= 1 )
          goto LABEL_21;
        if ( !*(_QWORD *)v111 )
        {
          v110 = 1;
          v9 = "Invalid vector length";
          goto LABEL_10;
        }
        v28 = sub_14EFEB0(v2, *((_QWORD *)v111 + 1));
        if ( !v28 || (v95 = v28, !(unsigned __int8)sub_1643C40(v28)) )
        {
LABEL_123:
          v110 = 1;
          v9 = "Invalid type";
          goto LABEL_10;
        }
        v12 = sub_16463B0(v95, *(_QWORD *)v111);
        goto LABEL_53;
      case 0xDu:
        v29 = sub_16432E0(v2[54]);
        v11 = v100;
        v12 = v29;
        goto LABEL_17;
      case 0xEu:
        v30 = sub_16432F0(v2[54]);
        v11 = v100;
        v12 = v30;
        goto LABEL_17;
      case 0xFu:
        v31 = sub_1643300(v2[54]);
        v11 = v100;
        v12 = v31;
        goto LABEL_17;
      case 0x10u:
        v32 = sub_16432C0(v2[54]);
        v11 = v100;
        v12 = v32;
        goto LABEL_17;
      case 0x11u:
        v33 = sub_1643310(v2[54]);
        v11 = v100;
        v12 = v33;
        goto LABEL_17;
      case 0x12u:
        if ( !(_DWORD)v112 )
          goto LABEL_21;
        v107 = &v109;
        v108 = 0x800000000LL;
        if ( (_DWORD)v112 == 1 )
        {
          v65 = 0;
        }
        else
        {
          v88 = v5;
          v34 = v112;
          v81 = v7;
          v35 = 1;
          do
          {
            v37 = sub_14EFEB0(v2, *(_QWORD *)&v111[8 * v35]);
            if ( !v37 )
            {
              v5 = v88;
              v7 = v81;
              v36 = v108;
              goto LABEL_107;
            }
            v38 = (unsigned int)v108;
            if ( (unsigned int)v108 >= HIDWORD(v108) )
            {
              v77 = v37;
              sub_16CD150(&v107, &v109, 0, 8);
              v38 = (unsigned int)v108;
              v37 = v77;
            }
            ++v35;
            *(_QWORD *)&v107[8 * v38] = v37;
            v36 = v108 + 1;
            LODWORD(v108) = v108 + 1;
          }
          while ( v35 != v34 );
          v5 = v88;
          v7 = v81;
LABEL_107:
          v65 = v36;
          if ( (unsigned int)v112 - 1LL != v36 )
            goto LABEL_103;
        }
        v66 = sub_1645600(v2[54], v107, v65, *(_QWORD *)v111 != 0);
        v63 = (unsigned __int64)v107;
        v12 = v66;
        if ( v107 == &v109 )
          goto LABEL_53;
        goto LABEL_99;
      case 0x13u:
        if ( (_DWORD)v112 )
        {
          v96 = v3;
          v39 = (unsigned int)v105;
          v89 = v2;
          v40 = v5;
          v82 = v6;
          v41 = (unsigned __int64)&v111[8 * (unsigned int)(v112 - 1) + 8];
          v42 = v7;
          v43 = v111;
          do
          {
            v44 = *(_QWORD *)v43;
            if ( HIDWORD(v105) <= (unsigned int)v39 )
            {
              sub_16CD150(&v104, v42, 0, 1);
              v39 = (unsigned int)v105;
            }
            v43 += 8;
            v104[v39] = v44;
            v39 = (unsigned int)(v105 + 1);
            LODWORD(v105) = v105 + 1;
          }
          while ( v43 != (const char *)v41 );
          v5 = v40;
          v7 = v42;
          v3 = v96;
          v2 = v89;
          v6 = v82;
        }
        continue;
      case 0x14u:
        if ( !(_DWORD)v112 )
          goto LABEL_21;
        v45 = v2[66];
        if ( v100 >= (unsigned __int64)((v2[67] - v45) >> 3) )
          goto LABEL_116;
        if ( *(_QWORD *)(v45 + 8LL * v100) )
        {
          v97 = *(_QWORD *)(v45 + 8LL * v100);
          sub_1643660(v97, v104, (unsigned int)v105);
          v46 = v100;
          v47 = v97;
          *(_QWORD *)(v2[66] + 8LL * v100) = 0;
        }
        else
        {
          v72 = sub_14F0050((__int64)v2, v2[54], (__int64)v104, (unsigned int)v105);
          v46 = v100;
          v47 = v72;
        }
        LODWORD(v105) = 0;
        v107 = &v109;
        v108 = 0x800000000LL;
        if ( (_DWORD)v112 == 1 )
        {
          v67 = 0;
        }
        else
        {
          v83 = v5;
          v48 = v112;
          v75 = v7;
          v49 = 1;
          v90 = v47;
          v78 = v46;
          do
          {
            v51 = sub_14EFEB0(v2, *(_QWORD *)&v111[8 * v49]);
            if ( !v51 )
            {
              v47 = v90;
              v5 = v83;
              v46 = v78;
              v7 = v75;
              v50 = v108;
              goto LABEL_111;
            }
            v52 = (unsigned int)v108;
            if ( (unsigned int)v108 >= HIDWORD(v108) )
            {
              v74 = v51;
              sub_16CD150(&v107, &v109, 0, 8);
              v52 = (unsigned int)v108;
              v51 = v74;
            }
            ++v49;
            *(_QWORD *)&v107[8 * v52] = v51;
            v50 = v108 + 1;
            LODWORD(v108) = v108 + 1;
          }
          while ( v49 != v48 );
          v47 = v90;
          v5 = v83;
          v46 = v78;
          v7 = v75;
LABEL_111:
          v67 = v50;
          if ( (unsigned int)v112 - 1LL != v50 )
          {
            v103 = 1;
            v64 = "Invalid record";
            goto LABEL_104;
          }
        }
        v85 = v46;
        v92 = v47;
        sub_1643FB0(v47, v107, v67, *(_QWORD *)v111 != 0);
        v12 = v92;
        v11 = v85;
        if ( v107 != &v109 )
        {
          _libc_free((unsigned __int64)v107);
          v12 = v92;
          v11 = v85;
        }
        goto LABEL_17;
      case 0x15u:
        if ( (unsigned int)v112 <= 1 )
        {
LABEL_21:
          v110 = 1;
          v9 = "Invalid record";
          goto LABEL_10;
        }
        v107 = &v109;
        v108 = 0x800000000LL;
        if ( (_DWORD)v112 != 2 )
        {
          v79 = v5;
          v53 = v112;
          v91 = v3;
          v54 = 2;
          v84 = v7;
          do
          {
            v55 = sub_14EFEB0(v2, *(_QWORD *)&v111[8 * v54]);
            v56 = v55;
            if ( !v55 )
              break;
            if ( !(unsigned __int8)sub_1643480(v55) )
            {
              v103 = 1;
              v3 = v91;
              v64 = "Invalid function argument type";
              v7 = v84;
              goto LABEL_104;
            }
            v57 = (unsigned int)v108;
            if ( (unsigned int)v108 >= HIDWORD(v108) )
            {
              sub_16CD150(&v107, &v109, 0, 8);
              v57 = (unsigned int)v108;
            }
            ++v54;
            *(_QWORD *)&v107[8 * v57] = v56;
            LODWORD(v108) = v108 + 1;
          }
          while ( v54 != v53 );
          v5 = v79;
          v3 = v91;
          v7 = v84;
        }
        v60 = sub_14EFEB0(v2, *((_QWORD *)v111 + 1));
        if ( v60 )
        {
          v61 = (unsigned int)v108;
          if ( (unsigned __int64)(unsigned int)v112 - 2 <= (unsigned int)v108 )
          {
LABEL_98:
            v62 = sub_1644EA0(v60, v107, v61, *(_QWORD *)v111 != 0);
            v63 = (unsigned __int64)v107;
            v12 = v62;
            if ( v107 != &v109 )
            {
LABEL_99:
              v99 = v12;
              _libc_free(v63);
              v12 = v99;
            }
LABEL_53:
            v11 = v100;
LABEL_17:
            v13 = v2[66];
            if ( (v2[67] - v13) >> 3 <= v11 )
            {
LABEL_116:
              v110 = 1;
              v9 = "Invalid TYPE table";
              goto LABEL_10;
            }
            v14 = (__int64 *)(v13 + 8 * v11);
            if ( *v14 )
            {
              v110 = 1;
              v9 = "Invalid TYPE table: Only named structs can be forward referenced";
              goto LABEL_10;
            }
            ++v100;
            *v14 = v12;
            continue;
          }
        }
LABEL_103:
        v103 = 1;
        v64 = "Invalid type";
LABEL_104:
        v101 = v64;
        v102 = 3;
        sub_14EE4B0(v3, (__int64)(v2 + 1), (__int64)&v101);
        if ( v107 != &v109 )
          _libc_free((unsigned __int64)v107);
LABEL_11:
        if ( v104 != v7 )
          _libc_free((unsigned __int64)v104);
        if ( v111 != v6 )
          _libc_free((unsigned __int64)v111);
        return v3;
      case 0x16u:
        v10 = sub_16432D0(v2[54]);
        v11 = v100;
        v12 = v10;
        goto LABEL_17;
      default:
        v110 = 1;
        v9 = "Invalid value";
        goto LABEL_10;
    }
  }
}
