// Function: sub_36D8FE0
// Address: 0x36d8fe0
//
char __fastcall sub_36D8FE0(__int64 a1, int a2)
{
  _DWORD *v3; // rax
  _DWORD *v4; // rdi
  __int64 (__fastcall *v5)(__int64); // rax
  __int64 v6; // rdi
  _DWORD *v7; // rdi
  __int64 (__fastcall *v8)(__int64); // rax
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 (*v11)(void); // rax
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 (__fastcall *v14)(__int64); // rax
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 (*v17)(void); // rax
  __int64 v18; // rdi
  _DWORD *v19; // rdi
  __int64 (__fastcall *v20)(__int64); // rax
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 (*v23)(void); // rax
  __int64 v24; // rdi
  _DWORD *v25; // rdi
  __int64 (__fastcall *v26)(__int64); // rax
  __int64 v27; // rdi
  __int64 v28; // rdi
  __int64 (*v29)(void); // rax
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 (*v32)(void); // rax
  __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 (*v35)(void); // rax
  __int64 v36; // rdi
  __int64 v37; // rdi
  __int64 (*v38)(void); // rax
  __int64 v39; // rdi
  __int64 v40; // rdi
  __int64 (*v41)(void); // rax
  __int64 v42; // rdi
  __int64 v43; // rdi
  __int64 (*v44)(void); // rax
  __int64 v45; // rdi
  __int64 v46; // rdi
  __int64 (*v47)(void); // rax
  __int64 v48; // rdi
  __int64 v49; // rdi
  __int64 (*v50)(void); // rax
  __int64 v51; // rdi
  __int64 v52; // rdi
  __int64 (*v53)(void); // rax
  __int64 v54; // rdi
  __int64 v55; // rdi
  __int64 (*v56)(void); // rax
  __int64 v57; // rdi
  __int64 v58; // rdi
  __int64 (*v59)(void); // rax
  __int64 v60; // rdi
  __int64 v61; // rdi
  __int64 (*v62)(void); // rax
  __int64 v63; // rdi
  __int64 v64; // rdi
  __int64 (*v65)(void); // rax
  __int64 v66; // rdi
  __int64 v67; // rdi
  __int64 (*v68)(void); // rax
  __int64 v69; // rdi
  __int64 v70; // rcx
  unsigned int v71; // edx
  __int64 v72; // rdi
  __int64 (*v73)(void); // rax
  __int64 v74; // rdi
  __int64 v75; // rdi
  unsigned int v76; // edx
  unsigned int v77; // ecx
  unsigned int v78; // ecx
  unsigned int v79; // edx
  __int64 v80; // rcx
  unsigned int v81; // edx
  __int64 v82; // rdi
  __int64 (*v83)(void); // rax
  __int64 v84; // rdi
  unsigned int v85; // edx
  unsigned int v86; // ecx
  __int64 v87; // rdi
  __int64 v88; // rdi
  __int64 (*v89)(void); // rax
  __int64 v90; // rdi
  __int64 v91; // rdx
  __int64 v92; // rdi
  __int64 (*v93)(void); // rax
  __int64 v94; // rdi
  __int64 v95; // rdi
  __int64 (*v96)(void); // rax
  __int64 v97; // rdi

  switch ( a2 )
  {
    case 0:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x31Fu )
        goto LABEL_26;
      LOBYTE(v3) = v3[84] > 0x45u;
      return (char)v3;
    case 1:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] <= 0x3Eu )
        goto LABEL_26;
      LOBYTE(v3) = v3[85] > 0x2CFu;
      return (char)v3;
    case 2:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x2BBu )
        goto LABEL_26;
      LOBYTE(v3) = v3[84] > 0x3Cu;
      return (char)v3;
    case 3:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] <= 0x3Bu )
        goto LABEL_26;
      LOBYTE(v3) = v3[85] > 0x12Bu;
      return (char)v3;
    case 4:
      v19 = *(_DWORD **)(a1 + 1136);
      v20 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v19 + 144LL);
      if ( v20 != sub_3020010 )
        goto LABEL_80;
      goto LABEL_40;
    case 5:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] <= 0x3Bu )
        goto LABEL_26;
      LOBYTE(v3) = v3[85] > 0x2BBu;
      return (char)v3;
    case 6:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] > 0x2BBu )
        goto LABEL_157;
      LOBYTE(v3) = 0;
      return (char)v3;
    case 7:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x2EDu )
        goto LABEL_26;
LABEL_157:
      LOBYTE(v3) = v3[84] > 0x3Eu;
      return (char)v3;
    case 8:
    case 22:
    case 35:
    case 45:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) > 0x3Bu;
      return (char)v3;
    case 9:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] > 0x55u )
        goto LABEL_50;
      LOBYTE(v3) = 0;
      return (char)v3;
    case 10:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] > 0x4Fu )
        goto LABEL_163;
      goto LABEL_26;
    case 11:
      v80 = *(_QWORD *)(a1 + 1136);
      v81 = *(_DWORD *)(v80 + 340);
      if ( v81 <= 0x12B )
        goto LABEL_26;
      LOBYTE(v3) = 1;
      if ( v81 > 0x2BB )
        LOBYTE(v3) = *(_DWORD *)(v80 + 336) <= 0x3Fu;
      return (char)v3;
    case 12:
      v75 = *(_QWORD *)(a1 + 1136);
      v76 = *(_DWORD *)(v75 + 340);
      if ( v76 > 0x408 )
      {
        LOBYTE(v3) = 0;
        if ( v76 - 1101 > 1 )
          return (char)v3;
      }
      else
      {
        LOBYTE(v3) = 0;
        if ( v76 <= 0x3E8 || ((1LL << ((unsigned __int8)v76 + 23)) & 0xC0000C03) == 0 )
          return (char)v3;
      }
      v77 = *(_DWORD *)(v75 + 336);
      if ( __ROR4__(-858993459 * v76 + 1717986918, 1) > 0x19999999u || (LOBYTE(v3) = 1, v77 <= 0x57) )
        LOBYTE(v3) = v77 > 0x55;
      return (char)v3;
    case 13:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x383u )
        goto LABEL_26;
      LOBYTE(v3) = v3[84] > 0x4Du;
      return (char)v3;
    case 14:
      v4 = *(_DWORD **)(a1 + 1136);
      v5 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 144LL);
      if ( v5 != sub_3020010 )
        goto LABEL_129;
      goto LABEL_11;
    case 15:
    case 87:
      goto LABEL_13;
    case 16:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 340LL) > 0x13Fu;
      return (char)v3;
    case 17:
      v7 = *(_DWORD **)(a1 + 1136);
      v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 144LL);
      if ( v8 != sub_3020010 )
        goto LABEL_240;
      goto LABEL_17;
    case 18:
    case 88:
      goto LABEL_19;
    case 19:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x383u )
        goto LABEL_26;
      LOBYTE(v3) = v3[84] > 0x55u;
      return (char)v3;
    case 20:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] > 0x40u )
        goto LABEL_103;
      goto LABEL_26;
    case 21:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] <= 0x47u )
        goto LABEL_26;
      LOBYTE(v3) = v3[85] > 0x35Bu;
      return (char)v3;
    case 23:
      v3 = *(_DWORD **)(a1 + 1136);
      v85 = v3[84];
      if ( v85 <= 0x55 )
        goto LABEL_26;
      LODWORD(v3) = v3[85];
      if ( (unsigned int)v3 <= 0x3E7 )
        goto LABEL_26;
      v86 = (unsigned int)v3 % 0xA;
      LOBYTE(v3) = 1;
      if ( v86 != 1 )
        LOBYTE(v3) = v86 == 2 && v85 > 0x57;
      return (char)v3;
    case 24:
    case 29:
    case 31:
    case 37:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) > 0x13u;
      return (char)v3;
    case 25:
      LOBYTE(v3) = *(_BYTE *)(a1 + 960);
      return (char)v3;
    case 26:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 952) + 648LL) != 0;
      return (char)v3;
    case 27:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] <= 0x55u )
        goto LABEL_26;
      LODWORD(v3) = v3[85];
      if ( (unsigned int)v3 <= 0x3E7 )
        goto LABEL_26;
      LOBYTE(v3) = __ROR4__(-858993459 * (_DWORD)v3 + 858993459, 1) <= 0x19999999u;
      return (char)v3;
    case 28:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) > 0xAu;
      return (char)v3;
    case 30:
    case 36:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) > 0xBu;
      return (char)v3;
    case 32:
      v92 = *(_QWORD *)(a1 + 1136);
      v93 = *(__int64 (**)(void))(*(_QWORD *)v92 + 144LL);
      if ( (char *)v93 == (char *)sub_3020010 )
        v94 = v92 + 960;
      else
        v94 = v93();
      LOBYTE(v3) = sub_3046050(v94, *(__int64 **)(a1 + 40), *(_DWORD *)(a1 + 792));
      return (char)v3;
    case 33:
      v25 = *(_DWORD **)(a1 + 1136);
      v26 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v25 + 144LL);
      if ( v26 != sub_3020010 )
        goto LABEL_60;
      goto LABEL_53;
    case 34:
      LOBYTE(v3) = 1;
      return (char)v3;
    case 38:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) > 0x59u;
      return (char)v3;
    case 39:
      v3 = *(_DWORD **)(a1 + 1136);
LABEL_163:
      LOBYTE(v3) = v3[85] > 0x383u;
      return (char)v3;
    case 40:
      v91 = *(_QWORD *)(a1 + 1136);
      if ( *(_DWORD *)(v91 + 344) <= 0x3Cu )
        goto LABEL_26;
      LOBYTE(v3) = 1;
      if ( *(_DWORD *)(v91 + 336) <= 0x31u )
        goto LABEL_26;
      return (char)v3;
    case 41:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x2BBu )
        goto LABEL_26;
      LOBYTE(v3) = v3[84] > 0x3Fu;
      return (char)v3;
    case 42:
      v88 = *(_QWORD *)(a1 + 1136);
      v89 = *(__int64 (**)(void))(*(_QWORD *)v88 + 144LL);
      if ( (char *)v89 == (char *)sub_3020010 )
        v90 = v88 + 960;
      else
        v90 = v89();
      if ( !sub_3037D80(v90, *(__int64 **)(a1 + 40)) )
        goto LABEL_26;
      v25 = *(_DWORD **)(a1 + 1136);
      v26 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v25 + 144LL);
      if ( v26 != sub_3020010 )
        goto LABEL_60;
      goto LABEL_53;
    case 43:
      v87 = *(_QWORD *)(a1 + 1136);
      if ( *(_DWORD *)(v87 + 340) > 0x31Fu && *(_DWORD *)(v87 + 336) > 0x45u )
        goto LABEL_205;
      goto LABEL_26;
    case 44:
    case 92:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x31Fu || v3[84] <= 0x45u )
        goto LABEL_26;
      LOBYTE(v3) = v3[86] > 0x4Fu;
      return (char)v3;
    case 46:
    case 47:
    case 48:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) > 0x45u;
      return (char)v3;
    case 49:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] <= 0x55u )
        goto LABEL_26;
      LOBYTE(v3) = v3[85] == 1001;
      return (char)v3;
    case 50:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] > 0x383u )
        goto LABEL_195;
      LOBYTE(v3) = 0;
      return (char)v3;
    case 51:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x211u )
        goto LABEL_26;
      LOBYTE(v3) = v3[84] > 0x29u;
      return (char)v3;
    case 52:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 336LL) > 0x2Au;
      return (char)v3;
    case 53:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) > 0x1Fu;
      return (char)v3;
    case 54:
      v10 = *(_QWORD *)(a1 + 1136);
      v11 = *(__int64 (**)(void))(*(_QWORD *)v10 + 144LL);
      if ( (char *)v11 == (char *)sub_3020010 )
        v12 = v10 + 960;
      else
        v12 = v11();
      if ( sub_3046050(v12, *(__int64 **)(a1 + 40), *(_DWORD *)(a1 + 792)) )
        goto LABEL_39;
      LOBYTE(v3) = 0;
      return (char)v3;
    case 55:
      v72 = *(_QWORD *)(a1 + 1136);
      v73 = *(__int64 (**)(void))(*(_QWORD *)v72 + 144LL);
      if ( (char *)v73 == (char *)sub_3020010 )
        v74 = v72 + 960;
      else
        v74 = v73();
      LOBYTE(v3) = sub_3037D80(v74, *(__int64 **)(a1 + 40)) ^ 1;
      return (char)v3;
    case 56:
      v70 = *(_QWORD *)(a1 + 1136);
      LODWORD(v3) = *(_DWORD *)(v70 + 340);
      if ( (unsigned int)v3 <= 0x3E7 )
        goto LABEL_26;
      v71 = (unsigned int)v3 % 0xA;
      LOBYTE(v3) = 1;
      if ( v71 == 1 )
        return (char)v3;
      if ( v71 != 2 )
        goto LABEL_26;
      LOBYTE(v3) = *(_DWORD *)(v70 + 336) > 0x57u;
      return (char)v3;
    case 57:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] <= 0x48u )
        goto LABEL_26;
      LOBYTE(v3) = v3[85] > 0x207u;
      return (char)v3;
    case 58:
      v64 = *(_QWORD *)(a1 + 1136);
      v65 = *(__int64 (**)(void))(*(_QWORD *)v64 + 144LL);
      if ( (char *)v65 == (char *)sub_3020010 )
        v66 = v64 + 960;
      else
        v66 = v65();
      if ( !sub_3046050(v66, *(__int64 **)(a1 + 40), *(_DWORD *)(a1 + 792)) )
        goto LABEL_26;
      v4 = *(_DWORD **)(a1 + 1136);
      v5 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 144LL);
      if ( v5 != sub_3020010 )
        goto LABEL_129;
      goto LABEL_11;
    case 59:
      v67 = *(_QWORD *)(a1 + 1136);
      v68 = *(__int64 (**)(void))(*(_QWORD *)v67 + 144LL);
      if ( (char *)v68 == (char *)sub_3020010 )
        v69 = v67 + 960;
      else
        v69 = v68();
      if ( sub_3046050(v69, *(__int64 **)(a1 + 40), *(_DWORD *)(a1 + 792)) )
        goto LABEL_13;
      LOBYTE(v3) = 0;
      return (char)v3;
    case 60:
      v46 = *(_QWORD *)(a1 + 1136);
      v47 = *(__int64 (**)(void))(*(_QWORD *)v46 + 144LL);
      if ( (char *)v47 == (char *)sub_3020010 )
        v48 = v46 + 960;
      else
        v48 = v47();
      if ( !sub_3037D80(v48, *(__int64 **)(a1 + 40)) )
        goto LABEL_26;
      v43 = *(_QWORD *)(a1 + 1136);
      v44 = *(__int64 (**)(void))(*(_QWORD *)v43 + 144LL);
      if ( (char *)v44 == (char *)sub_3020010 )
LABEL_82:
        v45 = v43 + 960;
      else
LABEL_89:
        v45 = v44();
      if ( !sub_3046050(v45, *(__int64 **)(a1 + 40), *(_DWORD *)(a1 + 792)) )
        goto LABEL_13;
      LOBYTE(v3) = 0;
      return (char)v3;
    case 61:
      v43 = *(_QWORD *)(a1 + 1136);
      v44 = *(__int64 (**)(void))(*(_QWORD *)v43 + 144LL);
      if ( (char *)v44 == (char *)sub_3020010 )
        goto LABEL_82;
      goto LABEL_89;
    case 62:
      v49 = *(_QWORD *)(a1 + 1136);
      v50 = *(__int64 (**)(void))(*(_QWORD *)v49 + 144LL);
      if ( (char *)v50 == (char *)sub_3020010 )
        v51 = v49 + 960;
      else
        v51 = v50();
      if ( sub_3046050(v51, *(__int64 **)(a1 + 40), *(_DWORD *)(a1 + 792)) )
        goto LABEL_19;
      goto LABEL_26;
    case 63:
      v28 = *(_QWORD *)(a1 + 1136);
      v29 = *(__int64 (**)(void))(*(_QWORD *)v28 + 144LL);
      if ( (char *)v29 == (char *)sub_3020010 )
        v30 = v28 + 960;
      else
        v30 = v29();
      if ( !sub_3037D80(v30, *(__int64 **)(a1 + 40)) )
        goto LABEL_26;
      v25 = *(_DWORD **)(a1 + 1136);
      if ( v25[86] <= 0x4Fu )
        goto LABEL_26;
      v26 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v25 + 144LL);
      if ( v26 != sub_3020010 )
        goto LABEL_60;
      goto LABEL_53;
    case 64:
      v25 = *(_DWORD **)(a1 + 1136);
      if ( v25[86] <= 0x4Fu )
        goto LABEL_26;
      v26 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v25 + 144LL);
      if ( v26 == sub_3020010 )
LABEL_53:
        v27 = (__int64)(v25 + 240);
      else
LABEL_60:
        v27 = ((__int64 (*)(void))v26)();
      LOBYTE(v3) = sub_3046050(v27, *(__int64 **)(a1 + 40), *(_DWORD *)(a1 + 792)) ^ 1;
      return (char)v3;
    case 65:
      v22 = *(_QWORD *)(a1 + 1136);
      v23 = *(__int64 (**)(void))(*(_QWORD *)v22 + 144LL);
      if ( (char *)v23 == (char *)sub_3020010 )
        v24 = v22 + 960;
      else
        v24 = v23();
      LOBYTE(v3) = sub_3046000(v24, *(__int64 **)(a1 + 40));
      return (char)v3;
    case 66:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x31Fu )
        goto LABEL_26;
      LOBYTE(v3) = v3[84] > 0x49u;
      return (char)v3;
    case 67:
      v3 = *(_DWORD **)(a1 + 1136);
LABEL_50:
      LOBYTE(v3) = v3[85] > 0x3E7u;
      return (char)v3;
    case 68:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x379u )
        goto LABEL_26;
      LOBYTE(v3) = v3[84] > 0x56u;
      return (char)v3;
    case 69:
      v4 = *(_DWORD **)(a1 + 1136);
      if ( v4[85] > 0x31Fu && v4[84] > 0x45u )
        goto LABEL_10;
      goto LABEL_26;
    case 70:
      v13 = *(_QWORD *)(a1 + 1136);
      v14 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 144LL);
      if ( v14 == sub_3020010 )
        goto LABEL_32;
      goto LABEL_234;
    case 71:
      v19 = *(_DWORD **)(a1 + 1136);
      if ( v19[85] <= 0x211u || v19[84] <= 0x40u )
        goto LABEL_26;
      v20 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v19 + 144LL);
      if ( v20 != sub_3020010 )
        goto LABEL_80;
      goto LABEL_40;
    case 72:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x211u )
        goto LABEL_26;
      LOBYTE(v3) = v3[84] > 0x40u;
      return (char)v3;
    case 73:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] <= 0x4Fu )
        goto LABEL_26;
      LOBYTE(v3) = v3[85] == 901;
      return (char)v3;
    case 74:
    case 96:
    case 101:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x31Fu )
        goto LABEL_26;
      LOBYTE(v3) = v3[84] > 0x46u;
      return (char)v3;
    case 75:
      v40 = *(_QWORD *)(a1 + 1136);
      v41 = *(__int64 (**)(void))(*(_QWORD *)v40 + 144LL);
      if ( (char *)v41 == (char *)sub_3020010 )
        v42 = v40 + 960;
      else
        v42 = v41();
      if ( !sub_3037D80(v42, *(__int64 **)(a1 + 40)) )
        goto LABEL_26;
      v31 = *(_QWORD *)(a1 + 1136);
      v32 = *(__int64 (**)(void))(*(_QWORD *)v31 + 144LL);
      if ( (char *)v32 == (char *)sub_3020010 )
LABEL_62:
        v33 = v31 + 960;
      else
LABEL_74:
        v33 = v32();
      LOBYTE(v3) = (unsigned int)sub_3037D10(v33) == 0;
      return (char)v3;
    case 76:
      v31 = *(_QWORD *)(a1 + 1136);
      v32 = *(__int64 (**)(void))(*(_QWORD *)v31 + 144LL);
      if ( (char *)v32 == (char *)sub_3020010 )
        goto LABEL_62;
      goto LABEL_74;
    case 77:
      v34 = *(_QWORD *)(a1 + 1136);
      v35 = *(__int64 (**)(void))(*(_QWORD *)v34 + 144LL);
      if ( (char *)v35 == (char *)sub_3020010 )
        v36 = v34 + 960;
      else
        v36 = v35();
      if ( !sub_3037D80(v36, *(__int64 **)(a1 + 40)) )
        goto LABEL_26;
      v37 = *(_QWORD *)(a1 + 1136);
      v38 = *(__int64 (**)(void))(*(_QWORD *)v37 + 144LL);
      if ( (char *)v38 == (char *)sub_3020010 )
LABEL_68:
        v39 = v37 + 960;
      else
LABEL_179:
        v39 = v38();
      LOBYTE(v3) = (unsigned int)sub_3037D10(v39) == 1;
      return (char)v3;
    case 78:
      v37 = *(_QWORD *)(a1 + 1136);
      v38 = *(__int64 (**)(void))(*(_QWORD *)v37 + 144LL);
      if ( (char *)v38 != (char *)sub_3020010 )
        goto LABEL_179;
      goto LABEL_68;
    case 79:
      v82 = *(_QWORD *)(a1 + 1136);
      v83 = *(__int64 (**)(void))(*(_QWORD *)v82 + 144LL);
      if ( (char *)v83 == (char *)sub_3020010 )
        v84 = v82 + 960;
      else
        v84 = v83();
      if ( (int)sub_3037D10(v84) <= 2 )
        goto LABEL_39;
      LOBYTE(v3) = 0;
      return (char)v3;
    case 80:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[85] <= 0x383u || v3[84] <= 0x4Du )
        goto LABEL_26;
      LOBYTE(v3) = v3[86] > 0x3Bu;
      return (char)v3;
    case 81:
      v3 = *(_DWORD **)(a1 + 1136);
      v78 = v3[84];
      if ( v78 <= 0x4F )
        goto LABEL_26;
      LODWORD(v3) = v3[85];
      if ( (unsigned int)v3 <= 0x383 )
        goto LABEL_26;
      v79 = (unsigned int)v3 % 0xA;
      LOBYTE(v3) = 1;
      if ( v79 != 1 )
        LOBYTE(v3) = v78 > 0x57 && v79 == 2;
      return (char)v3;
    case 82:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 336LL) > 0x1Eu;
      return (char)v3;
    case 83:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] > 0x45u )
LABEL_103:
        LOBYTE(v3) = v3[85] > 0x2EDu;
      else
        LOBYTE(v3) = 0;
      return (char)v3;
    case 84:
      if ( !(_BYTE)qword_5040E68 )
        goto LABEL_26;
      LOBYTE(v3) = qword_5040F48 ^ 1;
      return (char)v3;
    case 85:
      v61 = *(_QWORD *)(a1 + 1136);
      v62 = *(__int64 (**)(void))(*(_QWORD *)v61 + 144LL);
      if ( (char *)v62 == (char *)sub_3020010 )
        v63 = v61 + 960;
      else
        v63 = v62();
      if ( !sub_3037D80(v63, *(__int64 **)(a1 + 40)) )
        goto LABEL_110;
      LOBYTE(v3) = 0;
      return (char)v3;
    case 86:
      v58 = *(_QWORD *)(a1 + 1136);
      v59 = *(__int64 (**)(void))(*(_QWORD *)v58 + 144LL);
      if ( (char *)v59 == (char *)sub_3020010 )
        v60 = v58 + 960;
      else
        v60 = v59();
      if ( !sub_3037D80(v60, *(__int64 **)(a1 + 40)) )
        goto LABEL_26;
LABEL_110:
      if ( !(_BYTE)qword_5040E68 || (_BYTE)qword_5040F48 )
        goto LABEL_26;
      LOBYTE(v3) = !sub_36D8FA0(a1);
      return (char)v3;
    case 89:
      v16 = *(_QWORD *)(a1 + 1136);
      v17 = *(__int64 (**)(void))(*(_QWORD *)v16 + 144LL);
      if ( (char *)v17 == (char *)sub_3020010 )
        v18 = v16 + 960;
      else
        v18 = v17();
      if ( !(unsigned __int8)sub_3046000(v18, *(__int64 **)(a1 + 40)) )
        goto LABEL_26;
LABEL_39:
      v19 = *(_DWORD **)(a1 + 1136);
      v20 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v19 + 144LL);
      if ( v20 == sub_3020010 )
LABEL_40:
        v21 = (__int64)(v19 + 240);
      else
LABEL_80:
        v21 = ((__int64 (*)(void))v20)();
      LOBYTE(v3) = sub_3037D80(v21, *(__int64 **)(a1 + 40));
      return (char)v3;
    case 90:
      v95 = *(_QWORD *)(a1 + 1136);
      v96 = *(__int64 (**)(void))(*(_QWORD *)v95 + 144LL);
      if ( (char *)v96 == (char *)sub_3020010 )
        v97 = v95 + 960;
      else
        v97 = v96();
      if ( !sub_3037D80(v97, *(__int64 **)(a1 + 40)) )
        goto LABEL_26;
      v13 = *(_QWORD *)(a1 + 1136);
      v14 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 144LL);
      if ( v14 == sub_3020010 )
LABEL_32:
        v15 = v13 + 960;
      else
LABEL_234:
        v15 = ((__int64 (*)(void))v14)();
      LODWORD(v3) = sub_3046000(v15, *(__int64 **)(a1 + 40)) ^ 1;
      return (char)v3;
    case 91:
      v7 = *(_DWORD **)(a1 + 1136);
      if ( v7[85] <= 0x31Fu || v7[84] <= 0x45u )
        goto LABEL_26;
      v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 144LL);
      if ( v8 == sub_3020010 )
LABEL_17:
        v9 = (__int64)(v7 + 240);
      else
LABEL_240:
        v9 = ((__int64 (*)(void))v8)();
      if ( !sub_3037D80(v9, *(__int64 **)(a1 + 40)) )
        goto LABEL_26;
LABEL_19:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) > 0x4Fu;
      return (char)v3;
    case 93:
      v4 = *(_DWORD **)(a1 + 1136);
      if ( v4[85] <= 0x211u || v4[84] <= 0x3Bu )
        goto LABEL_26;
LABEL_10:
      v5 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 144LL);
      if ( v5 == sub_3020010 )
LABEL_11:
        v6 = (__int64)(v4 + 240);
      else
LABEL_129:
        v6 = ((__int64 (*)(void))v5)();
      if ( sub_3037D80(v6, *(__int64 **)(a1 + 40)) )
LABEL_13:
        LOBYTE(v3) = sub_305B500(*(_QWORD *)(a1 + 1136));
      else
LABEL_26:
        LOBYTE(v3) = 0;
      return (char)v3;
    case 94:
      v87 = *(_QWORD *)(a1 + 1136);
      if ( *(_DWORD *)(v87 + 340) <= 0x211u )
        goto LABEL_26;
      if ( *(_DWORD *)(v87 + 336) > 0x3Bu )
LABEL_205:
        LOBYTE(v3) = sub_305B500(v87);
      else
        LOBYTE(v3) = 0;
      return (char)v3;
    case 95:
      v3 = *(_DWORD **)(a1 + 1136);
      if ( v3[84] <= 0x50u )
        goto LABEL_26;
      LOBYTE(v3) = v3[85] > 0x379u;
      return (char)v3;
    case 97:
      v52 = *(_QWORD *)(a1 + 1136);
      v53 = *(__int64 (**)(void))(*(_QWORD *)v52 + 144LL);
      if ( (char *)v53 == (char *)sub_3020010 )
        v54 = v52 + 960;
      else
        v54 = v53();
      if ( !sub_3037D80(v54, *(__int64 **)(a1 + 40)) )
        goto LABEL_26;
      v55 = *(_QWORD *)(a1 + 1136);
      v56 = *(__int64 (**)(void))(*(_QWORD *)v55 + 144LL);
      if ( (char *)v56 == (char *)sub_3020010 )
LABEL_98:
        v57 = v55 + 960;
      else
LABEL_189:
        v57 = v56();
      LOBYTE(v3) = sub_3037D40(v57);
      return (char)v3;
    case 98:
      v55 = *(_QWORD *)(a1 + 1136);
      v56 = *(__int64 (**)(void))(*(_QWORD *)v55 + 144LL);
      if ( (char *)v56 != (char *)sub_3020010 )
        goto LABEL_189;
      goto LABEL_98;
    case 99:
      v3 = *(_DWORD **)(a1 + 1136);
LABEL_195:
      LOBYTE(v3) = v3[84] > 0x52u;
      return (char)v3;
    case 100:
      LOBYTE(v3) = *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 336LL) <= 0x52u;
      return (char)v3;
    default:
      BUG();
  }
}
