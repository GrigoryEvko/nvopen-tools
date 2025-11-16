// Function: sub_F96A40
// Address: 0xf96a40
//
__int64 __fastcall sub_F96A40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v7; // r14
  __int64 v11; // rax
  _BYTE *v12; // rax
  __int64 v13; // r10
  _QWORD *v14; // rsi
  __int64 v15; // rax
  bool v16; // r14
  bool v17; // dl
  unsigned int v18; // eax
  __int64 v19; // rcx
  __int64 v20; // r8
  unsigned int v21; // edi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // rdx
  __int64 *v34; // rax
  __int64 v35; // rsi
  __int64 v36; // r14
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 *v40; // rax
  __int64 v41; // r14
  __int64 *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // r9
  __int64 v47; // r10
  __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // r10
  __int64 v51; // r9
  char v52; // al
  int v53; // r13d
  __int64 v54; // rsi
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r10
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rsi
  _BYTE *v62; // rsi
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 *v66; // rax
  __int64 v67; // [rsp+18h] [rbp-108h]
  __int64 v68; // [rsp+18h] [rbp-108h]
  __int64 v69; // [rsp+18h] [rbp-108h]
  __int64 v70; // [rsp+18h] [rbp-108h]
  __int64 v71; // [rsp+20h] [rbp-100h]
  __int64 v72; // [rsp+20h] [rbp-100h]
  __int64 v73; // [rsp+20h] [rbp-100h]
  __int64 v74; // [rsp+20h] [rbp-100h]
  __int64 v75; // [rsp+28h] [rbp-F8h]
  __int64 v76; // [rsp+30h] [rbp-F0h]
  __int64 v77; // [rsp+38h] [rbp-E8h]
  __int64 v78; // [rsp+38h] [rbp-E8h]
  __int64 v79; // [rsp+40h] [rbp-E0h]
  __int64 v80; // [rsp+40h] [rbp-E0h]
  void **v81; // [rsp+48h] [rbp-D8h]
  __int64 v82; // [rsp+48h] [rbp-D8h]
  __int64 v83; // [rsp+60h] [rbp-C0h]
  unsigned __int64 v84; // [rsp+68h] [rbp-B8h]
  _BYTE *v85; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v86; // [rsp+78h] [rbp-A8h]
  _BYTE v87[32]; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v88[4]; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v89; // [rsp+C0h] [rbp-60h]
  __int64 v90; // [rsp+C8h] [rbp-58h]
  __int64 v91; // [rsp+D0h] [rbp-50h]
  __int64 v92; // [rsp+D8h] [rbp-48h]
  __int16 v93; // [rsp+E0h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(v3 + 56);
  if ( !v4 )
    BUG();
  if ( *(_BYTE *)(v4 - 24) == 84 )
    goto LABEL_4;
  v5 = *(_QWORD *)(a2 + 16);
  if ( !v5 )
    goto LABEL_4;
  if ( *(_QWORD *)(v5 + 8) )
    goto LABEL_4;
  v7 = *(_QWORD *)(a2 - 64);
  v81 = *(void ***)(a2 - 32);
  v11 = sub_AA54C0(*(_QWORD *)(a2 + 40));
  if ( !v11 )
    goto LABEL_4;
  v79 = v11;
  v12 = (_BYTE *)sub_986580(v11);
  v13 = (__int64)v12;
  if ( *v12 != 32 )
    goto LABEL_4;
  v14 = (_QWORD *)*((_QWORD *)v12 - 1);
  if ( v7 != *v14 )
    goto LABEL_4;
  v15 = v14[4];
  v16 = v15 == 0;
  v17 = v3 != v15;
  v18 = (*(_DWORD *)(v13 + 4) & 0x7FFFFFFu) >> 1;
  LOBYTE(v7) = v17 || v16;
  v19 = v18 - 1;
  if ( !(_BYTE)v7 )
  {
    v32 = v13;
    v77 = v18 - 1;
    sub_F90F90(v13, 0, v13, v19, (__int64)v81);
    if ( v33 != v77 && (_DWORD)v33 != -2 )
    {
      if ( (*(_WORD *)(a2 + 2) & 0x3F) == 0x20 )
      {
        v66 = (__int64 *)sub_AA48A0(v3);
        v35 = sub_ACD720(v66);
      }
      else
      {
        v34 = (__int64 *)sub_AA48A0(v3);
        v35 = sub_ACD6D0(v34);
      }
      LODWORD(v7) = 1;
      sub_BD84D0(a2, v35);
      sub_B43D60((_QWORD *)a2);
      *(_BYTE *)(a1 + 56) = 1;
      return (unsigned int)v7;
    }
    v75 = v79;
    v36 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL);
    v76 = v36;
    if ( *(_BYTE *)v36 == 84 )
    {
      v37 = sub_986580(v3);
      v78 = sub_B46EC0(v37, 0);
      v38 = *(_QWORD *)(v78 + 56);
      if ( v38 )
      {
        if ( v36 == v38 - 24 )
        {
          v39 = *(_QWORD *)(v36 + 32);
          if ( !v39 )
            BUG();
          if ( *(_BYTE *)(v39 - 24) != 84 )
          {
            v40 = (__int64 *)sub_AA48A0(v3);
            v41 = sub_ACD6D0(v40);
            v42 = (__int64 *)sub_AA48A0(v3);
            v80 = sub_ACD720(v42);
            if ( (*(_WORD *)(a2 + 2) & 0x3F) == 0x20 )
            {
              v43 = v41;
              v41 = v80;
              v80 = v43;
            }
            sub_BD84D0(a2, v41);
            sub_B43D60((_QWORD *)a2);
            v44 = *(_QWORD *)(v3 + 72);
            v85 = v87;
            v67 = v44;
            v86 = 0x200000000LL;
            v88[0] = (__int64)"switch.edge";
            LOWORD(v89) = 259;
            v71 = sub_AA48A0(v3);
            v45 = sub_22077B0(80);
            v46 = v75;
            v47 = v32;
            v48 = v45;
            if ( v45 )
            {
              sub_AA4D50(v45, v71, (__int64)v88, v67, v3);
              v47 = v32;
              v46 = v75;
            }
            v68 = v46;
            v88[0] = v47;
            v72 = v47;
            LOBYTE(v92) = 0;
            LOBYTE(v93) = 0;
            sub_B540B0(v88);
            v49 = sub_B543C0((__int64)v88, 0);
            v50 = v72;
            HIDWORD(v83) = 0;
            v51 = v68;
            if ( BYTE4(v49) )
            {
              BYTE4(v84) = 1;
              v70 = v72;
              v74 = v51;
              LODWORD(v84) = ((unsigned __int64)(unsigned int)v49 + 1) >> 1;
              v53 = v84;
              sub_B543F0((__int64)v88, 0, v84);
              v51 = v74;
              v52 = 1;
              v50 = v70;
            }
            else
            {
              v52 = 0;
              v53 = 0;
            }
            LODWORD(v83) = v53;
            v54 = (__int64)v81;
            BYTE4(v83) = v52;
            v69 = v50;
            v73 = v51;
            sub_B541D0((__int64)v88, v81, v48, v83);
            v57 = v69;
            if ( *(_QWORD *)(a1 + 8) )
            {
              v54 = v73;
              sub_F35FA0((__int64)&v85, v73, v48 & 0xFFFFFFFFFFFFFFFBLL, v55, v56, v73);
              v57 = v69;
            }
            v82 = v57;
            sub_F92F70((__int64)v88, v54);
            *(_QWORD *)(a3 + 48) = v48;
            *(_WORD *)(a3 + 64) = 0;
            *(_QWORD *)(a3 + 56) = v48 + 48;
            v61 = *(_QWORD *)(v82 + 48);
            v88[0] = v61;
            if ( v61 )
              sub_B96E90((__int64)v88, v61, 1);
            sub_F80810(a3, 0, v88[0], v58, v59, v60);
            sub_9C6650(v88);
            sub_F902B0((__int64 *)a3, v78);
            v62 = (_BYTE *)v80;
            sub_F0A850(v76, v80, v48);
            if ( *(_QWORD *)(a1 + 8) )
            {
              sub_F35FA0((__int64)&v85, v48, v78 & 0xFFFFFFFFFFFFFFFBLL, v63, v64, v65);
              v62 = v85;
              sub_FFB3D0(*(_QWORD *)(a1 + 8), v85, (unsigned int)v86);
            }
            if ( v85 != v87 )
              _libc_free(v85, v62);
            LODWORD(v7) = 1;
            return (unsigned int)v7;
          }
        }
      }
    }
LABEL_4:
    LODWORD(v7) = 0;
    return (unsigned int)v7;
  }
  v20 = 0;
  if ( v18 == 1 )
    goto LABEL_22;
  v21 = 2;
  v22 = 1;
  while ( 1 )
  {
    v24 = 4;
    if ( (_DWORD)v22 != -1 )
      v24 = 4LL * (v21 + 1);
    v25 = v14[v24];
    if ( !v25 || v3 != v25 )
    {
      v23 = v22;
      goto LABEL_16;
    }
    if ( v20 )
      break;
    v23 = v22;
    v20 = v14[4 * v21];
LABEL_16:
    ++v22;
    v21 += 2;
    if ( v19 == v23 )
      goto LABEL_22;
  }
  v20 = 0;
LABEL_22:
  sub_AC2B30(a2 - 64, v20);
  v26 = *(_QWORD *)(a1 + 24);
  memset(&v88[1], 0, 24);
  v88[0] = v26;
  v89 = 0;
  v90 = a2;
  v91 = 0;
  v92 = 0;
  v93 = 257;
  v31 = sub_1020E10(a2, v88, v27, v28, v29, v30);
  if ( v31 )
  {
    sub_BD84D0(a2, v31);
    sub_B43D60((_QWORD *)a2);
  }
  *(_BYTE *)(a1 + 56) = 1;
  return (unsigned int)v7;
}
