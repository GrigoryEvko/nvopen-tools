// Function: sub_27EE3E0
// Address: 0x27ee3e0
//
__int64 __fastcall sub_27EE3E0(unsigned __int8 *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  int v4; // r15d
  unsigned __int8 *v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned __int8 v8; // r14
  unsigned __int8 *v9; // r10
  char v10; // bl
  unsigned int v11; // r14d
  __int64 v13; // rdx
  __int64 v14; // rcx
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r10
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rcx
  _QWORD *v23; // rdx
  unsigned __int64 v24; // rax
  int v25; // edx
  __int64 v26; // r13
  __int64 v27; // rax
  __int64 v28; // r10
  __int64 (__fastcall *v29)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v30; // rax
  __int64 v31; // r15
  const char *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  _QWORD *v35; // r10
  unsigned __int8 *v36; // r13
  int v37; // ebx
  int v38; // eax
  __int64 v39; // r10
  int v40; // ebx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  char v47; // al
  unsigned __int8 *v48; // rax
  unsigned __int8 *v49; // rax
  __int64 v50; // r10
  int v51; // eax
  __int64 v52; // r13
  __int64 v53; // rcx
  int v54; // edx
  unsigned __int8 v55; // al
  bool v56; // al
  unsigned __int8 v57; // al
  int v58; // r13d
  __int64 v59; // r13
  _BYTE *v60; // rcx
  unsigned __int64 v61; // r13
  _BYTE *v62; // rbx
  __int64 v63; // rdx
  unsigned int v64; // esi
  bool v65; // al
  bool v66; // al
  __int64 v67; // rax
  __int64 v68; // [rsp+8h] [rbp-178h]
  __int64 v69; // [rsp+28h] [rbp-158h]
  unsigned __int8 *v70; // [rsp+30h] [rbp-150h]
  __int64 v71; // [rsp+30h] [rbp-150h]
  __int64 v72; // [rsp+38h] [rbp-148h]
  __int64 v73; // [rsp+38h] [rbp-148h]
  unsigned int v74; // [rsp+38h] [rbp-148h]
  unsigned __int8 *v75; // [rsp+40h] [rbp-140h]
  __int64 v76; // [rsp+40h] [rbp-140h]
  __int64 v77; // [rsp+40h] [rbp-140h]
  __int64 v78; // [rsp+40h] [rbp-140h]
  __int64 v79; // [rsp+40h] [rbp-140h]
  char v80; // [rsp+40h] [rbp-140h]
  __int64 v81; // [rsp+48h] [rbp-138h]
  __int64 v82; // [rsp+48h] [rbp-138h]
  _QWORD *v83; // [rsp+48h] [rbp-138h]
  _QWORD *v84; // [rsp+48h] [rbp-138h]
  _QWORD *v85; // [rsp+48h] [rbp-138h]
  const char *v88; // [rsp+60h] [rbp-120h] BYREF
  char v89; // [rsp+80h] [rbp-100h]
  char v90; // [rsp+81h] [rbp-FFh]
  _QWORD v91[4]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v92; // [rsp+B0h] [rbp-D0h]
  _BYTE *v93; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v94; // [rsp+C8h] [rbp-B8h]
  _BYTE v95[32]; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v96; // [rsp+F0h] [rbp-90h]
  __int64 v97; // [rsp+F8h] [rbp-88h]
  __int64 v98; // [rsp+100h] [rbp-80h]
  __int64 v99; // [rsp+108h] [rbp-78h]
  void **v100; // [rsp+110h] [rbp-70h]
  void **v101; // [rsp+118h] [rbp-68h]
  __int64 v102; // [rsp+120h] [rbp-60h]
  int v103; // [rsp+128h] [rbp-58h]
  __int16 v104; // [rsp+12Ch] [rbp-54h]
  char v105; // [rsp+12Eh] [rbp-52h]
  __int64 v106; // [rsp+130h] [rbp-50h]
  __int64 v107; // [rsp+138h] [rbp-48h]
  void *v108; // [rsp+140h] [rbp-40h] BYREF
  void *v109; // [rsp+148h] [rbp-38h] BYREF

  v4 = *a1;
  if ( (unsigned int)(v4 - 42) > 0x11 )
    return 0;
  v5 = a1;
  if ( !sub_B46CC0(a1) )
    return 0;
  v8 = sub_D48480(a2, *((_QWORD *)a1 - 8), v6, v7);
  v9 = *(unsigned __int8 **)&a1[32 * v8 - 64];
  if ( (unsigned __int8)(*v9 - 42) > 0x11u )
    return 0;
  v10 = v4;
  if ( (_BYTE)v4 != *v9 )
    return 0;
  v81 = *(_QWORD *)&a1[32 * v8 - 64];
  if ( !sub_B46CC0(v9) || (unsigned __int8)sub_BD3660(v81, (int)qword_4FFE248 + 1) )
    return 0;
  v72 = v81;
  v75 = *(unsigned __int8 **)(v81 - 32);
  v82 = *(_QWORD *)(v81 - 64);
  v70 = *(unsigned __int8 **)&a1[32 * (v8 ^ 1) - 64];
  v15 = sub_D48480(a2, v82, v13, v14);
  v18 = v72;
  if ( v15 )
  {
    v47 = sub_D48480(a2, (__int64)v75, v16, v17);
    v18 = v72;
    if ( !v47 )
    {
      v48 = (unsigned __int8 *)v82;
      v82 = (__int64)v75;
      v75 = v48;
    }
  }
  v73 = v18;
  if ( (unsigned __int8)sub_D48480(a2, v82, v16, v17) )
    return 0;
  if ( !(unsigned __int8)sub_D48480(a2, (__int64)v75, v19, v20) )
    return 0;
  v11 = sub_D48480(a2, (__int64)v70, v21, v22);
  if ( !(_BYTE)v11 )
    return 0;
  v23 = (_QWORD *)(sub_D4B130(a2) + 48);
  v24 = *v23 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v24 == v23 )
  {
    v26 = 0;
  }
  else
  {
    if ( !v24 )
      BUG();
    v25 = *(unsigned __int8 *)(v24 - 24);
    v26 = 0;
    v27 = v24 - 24;
    if ( (unsigned int)(v25 - 30) < 0xB )
      v26 = v27;
  }
  v69 = v73;
  v74 = v4 - 29;
  v105 = 7;
  v99 = sub_BD5C60(v26);
  v100 = &v108;
  v101 = &v109;
  v93 = v95;
  v108 = &unk_49DA100;
  v94 = 0x200000000LL;
  v102 = 0;
  v109 = &unk_49DA0B0;
  v103 = 0;
  v104 = 512;
  v106 = 0;
  v107 = 0;
  v96 = 0;
  v97 = 0;
  LOWORD(v98) = 0;
  sub_D5F1F0((__int64)&v93, v26);
  v90 = 1;
  v89 = 3;
  v28 = v69;
  v88 = "invariant.op";
  v29 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v100 + 2);
  if ( v29 == sub_9202E0 )
  {
    if ( *v75 > 0x15u || *v70 > 0x15u )
      goto LABEL_35;
    if ( (unsigned __int8)sub_AC47B0(v74) )
      v30 = sub_AD5570(v74, (__int64)v75, v70, 0, 0);
    else
      v30 = sub_AABE40(v74, v75, v70);
    v28 = v69;
    v31 = v30;
  }
  else
  {
    v67 = v29((__int64)v100, v74, v75, v70);
    v28 = v69;
    v31 = v67;
  }
  if ( !v31 )
  {
LABEL_35:
    v92 = 257;
    v68 = v28;
    v49 = (unsigned __int8 *)sub_B504D0(v74, (__int64)v75, (__int64)v70, (__int64)v91, 0, 0);
    v50 = v68;
    v31 = (__int64)v49;
    v51 = *v49;
    if ( (unsigned __int8)v51 > 0x1Cu )
    {
      switch ( v51 )
      {
        case ')':
        case '+':
        case '-':
        case '/':
        case '2':
        case '5':
        case 'J':
        case 'K':
        case 'S':
          goto LABEL_51;
        case 'T':
        case 'U':
        case 'V':
          v52 = *(_QWORD *)(v31 + 8);
          v53 = v52;
          v54 = *(unsigned __int8 *)(v52 + 8);
          if ( (unsigned int)(v54 - 17) <= 1 )
            v53 = **(_QWORD **)(v52 + 16);
          v55 = *(_BYTE *)(v53 + 8);
          if ( v55 <= 3u || v55 == 5 || (v55 & 0xFD) == 4 )
            goto LABEL_51;
          if ( (_BYTE)v54 == 15 )
          {
            if ( (*(_BYTE *)(v52 + 9) & 4) == 0 )
              break;
            v56 = sub_BCB420(*(_QWORD *)(v31 + 8));
            v50 = v68;
            if ( !v56 )
              break;
            v52 = **(_QWORD **)(v52 + 16);
          }
          else if ( (_BYTE)v54 == 16 )
          {
            do
              v52 = *(_QWORD *)(v52 + 24);
            while ( *(_BYTE *)(v52 + 8) == 16 );
          }
          if ( (unsigned int)*(unsigned __int8 *)(v52 + 8) - 17 <= 1 )
            v52 = **(_QWORD **)(v52 + 16);
          v57 = *(_BYTE *)(v52 + 8);
          if ( v57 <= 3u || v57 == 5 || (v57 & 0xFD) == 4 )
          {
LABEL_51:
            v58 = v103;
            if ( v102 )
            {
              v77 = v50;
              sub_B99FD0(v31, 3u, v102);
              v50 = v77;
            }
            v78 = v50;
            sub_B45150(v31, v58);
            v50 = v78;
          }
          break;
        default:
          break;
      }
    }
    v79 = v50;
    (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v101 + 2))(v101, v31, &v88, v97, v98);
    v28 = v79;
    v59 = 16LL * (unsigned int)v94;
    v60 = &v93[v59];
    if ( v93 != &v93[v59] )
    {
      v71 = v79;
      v61 = (unsigned __int64)v93;
      v80 = v10;
      v62 = v60;
      do
      {
        v63 = *(_QWORD *)(v61 + 8);
        v64 = *(_DWORD *)v61;
        v61 += 16LL;
        sub_B99FD0(v31, v64, v63);
      }
      while ( v62 != (_BYTE *)v61 );
      v10 = v80;
      v28 = v71;
      v5 = a1;
    }
  }
  v76 = v28;
  v32 = sub_BD5D20((__int64)v5);
  v92 = 773;
  v91[0] = v32;
  v91[1] = v33;
  v91[2] = ".reass";
  v34 = sub_B504D0(v74, v82, v31, (__int64)v91, (__int64)(v5 + 24), 0);
  v35 = (_QWORD *)v76;
  v36 = (unsigned __int8 *)v34;
  if ( v74 == 13 )
  {
    v65 = sub_B448F0((__int64)v5);
    v35 = (_QWORD *)v76;
    if ( v65 )
    {
      v66 = sub_B448F0(v76);
      v35 = (_QWORD *)v76;
      if ( v66 )
      {
        if ( *(_BYTE *)v31 > 0x1Cu )
        {
          sub_B447F0((unsigned __int8 *)v31, 1);
          v35 = (_QWORD *)v76;
        }
        v85 = v35;
        sub_B447F0(v36, 1);
        v35 = v85;
      }
    }
  }
  else if ( (v10 & 0xFB) == 0x2B )
  {
    v37 = sub_B45210(v76);
    v38 = sub_B45210((__int64)v5);
    v39 = v76;
    v40 = v38 & v37;
    if ( *(_BYTE *)v31 > 0x1Cu )
    {
      sub_B45150(v31, v40);
      v39 = v76;
    }
    v83 = (_QWORD *)v39;
    sub_B45150((__int64)v36, v40);
    v35 = v83;
  }
  v84 = v35;
  sub_BD84D0((__int64)v5, (__int64)v36);
  sub_27EC480(v5, a3, a4, v41, v42, v43);
  if ( !v84[2] )
    sub_27EC480(v84, a3, a4, v44, v45, v46);
  nullsub_61();
  v108 = &unk_49DA100;
  nullsub_63();
  if ( v93 != v95 )
    _libc_free((unsigned __int64)v93);
  return v11;
}
