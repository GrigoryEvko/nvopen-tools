// Function: sub_2A2C8D0
// Address: 0x2a2c8d0
//
__int64 __fastcall sub_2A2C8D0(int a1, __int64 a2, __int64 a3, unsigned __int8 *a4)
{
  __int64 v4; // rbp
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 (__fastcall *v18)(__int64, __int64, __int64); // rax
  unsigned int *v19; // rbx
  __int64 v20; // r12
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // rdi
  __int64 (__fastcall *v24)(__int64, __int64, __int64); // rax
  __int64 v25; // r10
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // r10
  unsigned __int8 *v29; // r13
  __int64 (__fastcall *v30)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v31; // rax
  unsigned int *v32; // rbx
  __int64 v33; // r12
  __int64 v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // rdi
  __int64 (__fastcall *v37)(__int64, __int64, __int64); // rax
  unsigned int *v38; // rbx
  __int64 v39; // r12
  __int64 v40; // rdx
  unsigned int v41; // esi
  __int64 v42; // rdi
  __int64 (__fastcall *v43)(__int64, __int64, __int64); // rax
  unsigned int *v44; // rbx
  __int64 v45; // r12
  __int64 v46; // rdx
  unsigned int v47; // esi
  unsigned int v48; // esi
  __int64 v49; // rax
  bool v50; // zf
  __int64 v51; // rdi
  unsigned int v52; // ebx
  __int64 (__fastcall *v53)(__int64, __int64, __int64, unsigned __int8 *, _QWORD); // rax
  __int64 v54; // rax
  __int64 v55; // rdx
  unsigned int *v56; // rbx
  __int64 v57; // r12
  __int64 v58; // rdx
  unsigned int v59; // esi
  __int64 v60; // rdi
  unsigned int v61; // ebx
  __int64 (__fastcall *v62)(__int64, __int64, __int64, unsigned __int8 *, _QWORD); // rax
  __int64 v63; // rax
  __int64 v64; // rdx
  unsigned int *v65; // rbx
  __int64 v66; // r12
  __int64 v67; // rdx
  unsigned int v68; // esi
  _BYTE *v69; // rax
  __int64 v70; // rax
  __int64 v71; // r14
  __int64 v72; // rax
  _BYTE *v73; // r14
  _BYTE *v74; // rax
  unsigned __int8 *v75; // r14
  __int64 v76; // rax
  __int64 v77; // rdi
  unsigned __int8 *v78; // r13
  __int64 (__fastcall *v79)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v80; // r11
  unsigned int *v81; // rbx
  __int64 v82; // r13
  __int64 v83; // rdx
  unsigned int v84; // esi
  unsigned int *v85; // r14
  __int64 v86; // r13
  __int64 v87; // rdx
  unsigned int v88; // esi
  __int64 v89; // rax
  __int64 v90; // [rsp-E8h] [rbp-E8h]
  unsigned __int8 *v91; // [rsp-E0h] [rbp-E0h]
  __int64 v92; // [rsp-E0h] [rbp-E0h]
  __int64 v93; // [rsp-E0h] [rbp-E0h]
  __int64 v94; // [rsp-E0h] [rbp-E0h]
  __int64 v95; // [rsp-E0h] [rbp-E0h]
  __int64 v96; // [rsp-E0h] [rbp-E0h]
  __int64 v97; // [rsp-D0h] [rbp-D0h] BYREF
  __int64 v98; // [rsp-C8h] [rbp-C8h] BYREF
  __int16 v99; // [rsp-A8h] [rbp-A8h]
  _QWORD v100[4]; // [rsp-98h] [rbp-98h] BYREF
  __int16 v101; // [rsp-78h] [rbp-78h]
  _QWORD v102[4]; // [rsp-68h] [rbp-68h] BYREF
  __int16 v103; // [rsp-48h] [rbp-48h]
  __int64 v104; // [rsp-28h] [rbp-28h]
  __int64 v105; // [rsp-20h] [rbp-20h]
  __int64 v106; // [rsp-10h] [rbp-10h]
  __int64 v107; // [rsp-8h] [rbp-8h]

  v107 = v4;
  v106 = v7;
  v105 = v6;
  v104 = v5;
  switch ( a1 )
  {
    case 0:
      return (__int64)a4;
    case 1:
      v102[0] = "new";
      v103 = 259;
      return sub_929C50((unsigned int **)a2, (_BYTE *)a3, a4, (__int64)v102, 0, 0);
    case 2:
      v102[0] = "new";
      v103 = 259;
      return sub_929DE0((unsigned int **)a2, (_BYTE *)a3, a4, (__int64)v102, 0, 0);
    case 3:
      v17 = *(_QWORD *)(a2 + 80);
      v100[0] = "new";
      v101 = 259;
      v18 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v17 + 16LL);
      if ( (char *)v18 == (char *)sub_9202E0 )
      {
        if ( *(_BYTE *)a3 > 0x15u || *a4 > 0x15u )
          goto LABEL_14;
        if ( (unsigned __int8)sub_AC47B0(28) )
          v14 = sub_AD5570(28, a3, a4, 0, 0);
        else
          v14 = sub_AABE40(0x1Cu, (unsigned __int8 *)a3, a4);
      }
      else
      {
        v14 = v18(v17, 28, a3);
      }
      if ( v14 )
        return v14;
LABEL_14:
      v103 = 257;
      v14 = sub_B504D0(28, a3, (__int64)a4, (__int64)v102, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v14,
        v100,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v19 = *(unsigned int **)a2;
      v20 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v20 )
      {
        do
        {
          v21 = *((_QWORD *)v19 + 1);
          v22 = *v19;
          v19 += 4;
          sub_B99FD0(v14, v22, v21);
        }
        while ( (unsigned int *)v20 != v19 );
      }
      return v14;
    case 4:
      v23 = *(_QWORD *)(a2 + 80);
      v100[0] = "new";
      v101 = 259;
      v99 = 257;
      v24 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v23 + 16LL);
      if ( (char *)v24 == (char *)sub_9202E0 )
      {
        if ( *(_BYTE *)a3 > 0x15u || *a4 > 0x15u )
          goto LABEL_103;
        if ( (unsigned __int8)sub_AC47B0(28) )
          v25 = sub_AD5570(28, a3, a4, 0, 0);
        else
          v25 = sub_AABE40(0x1Cu, (unsigned __int8 *)a3, a4);
      }
      else
      {
        v25 = v24(v23, 28, a3);
      }
      if ( v25 )
        goto LABEL_23;
LABEL_103:
      v103 = 257;
      v93 = sub_B504D0(28, a3, (__int64)a4, (__int64)v102, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v93,
        &v98,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v81 = *(unsigned int **)a2;
      v25 = v93;
      v82 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v82 )
      {
        do
        {
          v83 = *((_QWORD *)v81 + 1);
          v84 = *v81;
          v81 += 4;
          v94 = v25;
          sub_B99FD0(v25, v84, v83);
          v25 = v94;
        }
        while ( (unsigned int *)v82 != v81 );
      }
LABEL_23:
      v91 = (unsigned __int8 *)v25;
      v26 = sub_AD62B0(*(_QWORD *)(v25 + 8));
      v27 = *(_QWORD *)(a2 + 80);
      v28 = (__int64)v91;
      v29 = (unsigned __int8 *)v26;
      v30 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v27 + 16LL);
      if ( v30 == sub_9202E0 )
      {
        if ( *v91 > 0x15u || *v29 > 0x15u )
          goto LABEL_30;
        if ( (unsigned __int8)sub_AC47B0(30) )
          v31 = sub_AD5570(30, (__int64)v91, v29, 0, 0);
        else
          v31 = sub_AABE40(0x1Eu, v91, v29);
        v28 = (__int64)v91;
        v14 = v31;
      }
      else
      {
        v89 = v30(v27, 30u, v91, v29);
        v28 = (__int64)v91;
        v14 = v89;
      }
      if ( v14 )
        return v14;
LABEL_30:
      v103 = 257;
      v14 = sub_B504D0(30, v28, (__int64)v29, (__int64)v102, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v14,
        v100,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v32 = *(unsigned int **)a2;
      v33 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      while ( (unsigned int *)v33 != v32 )
      {
        v34 = *((_QWORD *)v32 + 1);
        v35 = *v32;
        v32 += 4;
        sub_B99FD0(v14, v35, v34);
      }
      return v14;
    case 5:
      v36 = *(_QWORD *)(a2 + 80);
      v100[0] = "new";
      v101 = 259;
      v37 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v36 + 16LL);
      if ( (char *)v37 == (char *)sub_9202E0 )
      {
        if ( *(_BYTE *)a3 > 0x15u || *a4 > 0x15u )
          goto LABEL_39;
        if ( (unsigned __int8)sub_AC47B0(29) )
          v14 = sub_AD5570(29, a3, a4, 0, 0);
        else
          v14 = sub_AABE40(0x1Du, (unsigned __int8 *)a3, a4);
      }
      else
      {
        v14 = v37(v36, 29, a3);
      }
      if ( v14 )
        return v14;
LABEL_39:
      v103 = 257;
      v14 = sub_B504D0(29, a3, (__int64)a4, (__int64)v102, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v14,
        v100,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v38 = *(unsigned int **)a2;
      v39 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v39 )
      {
        do
        {
          v40 = *((_QWORD *)v38 + 1);
          v41 = *v38;
          v38 += 4;
          sub_B99FD0(v14, v41, v40);
        }
        while ( (unsigned int *)v39 != v38 );
      }
      return v14;
    case 6:
      v42 = *(_QWORD *)(a2 + 80);
      v100[0] = "new";
      v101 = 259;
      v43 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v42 + 16LL);
      if ( (char *)v43 == (char *)sub_9202E0 )
      {
        if ( *(_BYTE *)a3 > 0x15u || *a4 > 0x15u )
          goto LABEL_48;
        if ( (unsigned __int8)sub_AC47B0(30) )
          v14 = sub_AD5570(30, a3, a4, 0, 0);
        else
          v14 = sub_AABE40(0x1Eu, (unsigned __int8 *)a3, a4);
      }
      else
      {
        v14 = v43(v42, 30, a3);
      }
      if ( v14 )
        return v14;
LABEL_48:
      v103 = 257;
      v14 = sub_B504D0(30, a3, (__int64)a4, (__int64)v102, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v14,
        v100,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v44 = *(unsigned int **)a2;
      v45 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v45 )
      {
        do
        {
          v46 = *((_QWORD *)v44 + 1);
          v47 = *v44;
          v44 += 4;
          sub_B99FD0(v14, v47, v46);
        }
        while ( (unsigned int *)v45 != v44 );
      }
      return v14;
    case 7:
      v48 = 38;
      v103 = 257;
      goto LABEL_52;
    case 8:
      v48 = 41;
      v103 = 257;
      goto LABEL_52;
    case 9:
      v48 = 34;
      v103 = 257;
      goto LABEL_52;
    case 10:
      v48 = 37;
      v103 = 257;
LABEL_52:
      v49 = sub_92B530((unsigned int **)a2, v48, a3, a4, (__int64)v102);
      v102[0] = "new";
      v103 = 259;
      return sub_B36550((unsigned int **)a2, v49, a3, (__int64)a4, (__int64)v102, 0);
    case 11:
      v50 = *(_BYTE *)(a2 + 108) == 0;
      HIDWORD(v97) = 0;
      v100[0] = "new";
      v101 = 259;
      v98 = (unsigned int)v97;
      if ( !v50 )
        return sub_B35400(a2, 0x66u, a3, (__int64)a4, (unsigned int)v97, (__int64)v100, 0, 0, 0);
      v51 = *(_QWORD *)(a2 + 80);
      v52 = *(_DWORD *)(a2 + 104);
      v53 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int8 *, _QWORD))(*(_QWORD *)v51 + 40LL);
      if ( (char *)v53 == (char *)sub_928A40 )
      {
        if ( *(_BYTE *)a3 > 0x15u || *a4 > 0x15u )
          goto LABEL_64;
        if ( (unsigned __int8)sub_AC47B0(14) )
          v14 = sub_AD5570(14, a3, a4, 0, 0);
        else
          v14 = sub_AABE40(0xEu, (unsigned __int8 *)a3, a4);
      }
      else
      {
        v14 = v53(v51, 14, a3, a4, v52);
      }
      if ( v14 )
        return v14;
      v52 = *(_DWORD *)(a2 + 104);
LABEL_64:
      v103 = 257;
      v54 = sub_B504D0(14, a3, (__int64)a4, (__int64)v102, 0, 0);
      v55 = *(_QWORD *)(a2 + 96);
      v14 = v54;
      if ( v55 )
        sub_B99FD0(v54, 3u, v55);
      sub_B45150(v14, v52);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v14,
        v100,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v56 = *(unsigned int **)a2;
      v57 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v57 )
      {
        do
        {
          v58 = *((_QWORD *)v56 + 1);
          v59 = *v56;
          v56 += 4;
          sub_B99FD0(v14, v59, v58);
        }
        while ( (unsigned int *)v57 != v56 );
      }
      return v14;
    case 12:
      v50 = *(_BYTE *)(a2 + 108) == 0;
      HIDWORD(v97) = 0;
      v100[0] = "new";
      v101 = 259;
      v98 = (unsigned int)v97;
      if ( !v50 )
        return sub_B35400(a2, 0x73u, a3, (__int64)a4, (unsigned int)v97, (__int64)v100, 0, 0, 0);
      v60 = *(_QWORD *)(a2 + 80);
      v61 = *(_DWORD *)(a2 + 104);
      v62 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int8 *, _QWORD))(*(_QWORD *)v60 + 40LL);
      if ( (char *)v62 == (char *)sub_928A40 )
      {
        if ( *(_BYTE *)a3 > 0x15u || *a4 > 0x15u )
          goto LABEL_77;
        if ( (unsigned __int8)sub_AC47B0(16) )
          v14 = sub_AD5570(16, a3, a4, 0, 0);
        else
          v14 = sub_AABE40(0x10u, (unsigned __int8 *)a3, a4);
      }
      else
      {
        v14 = v62(v60, 16, a3, a4, v61);
      }
      if ( v14 )
        return v14;
      v61 = *(_DWORD *)(a2 + 104);
LABEL_77:
      v103 = 257;
      v63 = sub_B504D0(16, a3, (__int64)a4, (__int64)v102, 0, 0);
      v64 = *(_QWORD *)(a2 + 96);
      v14 = v63;
      if ( v64 )
        sub_B99FD0(v63, 3u, v64);
      sub_B45150(v14, v61);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v14,
        v100,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v65 = *(unsigned int **)a2;
      v66 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v66 )
      {
        do
        {
          v67 = *((_QWORD *)v65 + 1);
          v68 = *v65;
          v65 += 4;
          sub_B99FD0(v14, v68, v67);
        }
        while ( (unsigned int *)v66 != v65 );
      }
      return v14;
    case 13:
      HIDWORD(v98) = 0;
      v50 = *(_BYTE *)(a2 + 108) == 0;
      v103 = 257;
      v100[0] = (unsigned int)v98;
      if ( v50 )
        return sub_B33C40(a2, 0xEDu, a3, (__int64)a4, (unsigned int)v98, (__int64)v102);
      else
        return sub_B35570(a2, 0x7Du, a3, (__int64)a4, (unsigned int)v98, (__int64)v102, 0, 0);
    case 14:
      HIDWORD(v98) = 0;
      v50 = *(_BYTE *)(a2 + 108) == 0;
      v103 = 257;
      v100[0] = (unsigned int)v98;
      if ( v50 )
        return sub_B33C40(a2, 0xF8u, a3, (__int64)a4, (unsigned int)v98, (__int64)v102);
      else
        return sub_B35570(a2, 0x7Fu, a3, (__int64)a4, (unsigned int)v98, (__int64)v102, 0, 0);
    case 15:
      v69 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a3 + 8), 1, 0);
      v103 = 257;
      v70 = sub_929C50((unsigned int **)a2, (_BYTE *)a3, v69, (__int64)v102, 0, 0);
      v103 = 257;
      v92 = v70;
      v71 = sub_92B530((unsigned int **)a2, 0x23u, a3, a4, (__int64)v102);
      v72 = sub_AD64C0(*(_QWORD *)(a3 + 8), 0, 0);
      v103 = 259;
      v102[0] = "new";
      return sub_B36550((unsigned int **)a2, v71, v72, v92, (__int64)v102, 0);
    case 16:
      v73 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a3 + 8), 0, 0);
      v74 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a3 + 8), 1, 0);
      v103 = 257;
      v90 = sub_929DE0((unsigned int **)a2, (_BYTE *)a3, v74, (__int64)v102, 0, 0);
      v103 = 257;
      v75 = (unsigned __int8 *)sub_92B530((unsigned int **)a2, 0x20u, a3, v73, (__int64)v102);
      v103 = 257;
      v76 = sub_92B530((unsigned int **)a2, 0x22u, a3, a4, (__int64)v102);
      v77 = *(_QWORD *)(a2 + 80);
      v101 = 257;
      v78 = (unsigned __int8 *)v76;
      v79 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v77 + 16LL);
      if ( v79 == sub_9202E0 )
      {
        if ( *v75 > 0x15u || *v78 > 0x15u )
          goto LABEL_106;
        if ( (unsigned __int8)sub_AC47B0(29) )
          v80 = sub_AD5570(29, (__int64)v75, v78, 0, 0);
        else
          v80 = sub_AABE40(0x1Du, v75, v78);
      }
      else
      {
        v80 = v79(v77, 29u, v75, v78);
      }
      if ( v80 )
        goto LABEL_93;
LABEL_106:
      v103 = 257;
      v95 = sub_B504D0(29, (__int64)v75, (__int64)v78, (__int64)v102, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v95,
        v100,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v85 = *(unsigned int **)a2;
      v80 = v95;
      v86 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v86 )
      {
        do
        {
          v87 = *((_QWORD *)v85 + 1);
          v88 = *v85;
          v85 += 4;
          v96 = v80;
          sub_B99FD0(v80, v88, v87);
          v80 = v96;
        }
        while ( (unsigned int *)v86 != v85 );
      }
LABEL_93:
      v102[0] = "new";
      v103 = 259;
      return sub_B36550((unsigned int **)a2, v80, (__int64)a4, v90, (__int64)v102, 0);
    case 17:
      v103 = 257;
      v11 = sub_92B530((unsigned int **)a2, 0x23u, a3, a4, (__int64)v102);
      v103 = 257;
      v12 = v11;
      v13 = sub_929DE0((unsigned int **)a2, (_BYTE *)a3, a4, (__int64)v102, 0, 0);
      v102[0] = "new";
      v103 = 259;
      return sub_B36550((unsigned int **)a2, v12, v13, a3, (__int64)v102, 0);
    case 18:
      BYTE4(v98) = 0;
      v102[0] = "new";
      v16 = *(_QWORD *)(a3 + 8);
      v100[1] = a4;
      v97 = v16;
      v103 = 259;
      v100[0] = a3;
      return sub_B33D10(a2, 0x173u, (__int64)&v97, 1, (int)v100, 2, v98, (__int64)v102);
    default:
      BUG();
  }
}
