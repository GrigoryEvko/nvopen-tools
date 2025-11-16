// Function: sub_115C220
// Address: 0x115c220
//
unsigned __int8 *__fastcall sub_115C220(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rax
  unsigned __int64 v5; // rsi
  __int64 v6; // r15
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  bool v10; // al
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rcx
  unsigned __int8 v15; // bl
  void *v16; // rbx
  void *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  unsigned __int8 *v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r13
  __int64 v28; // rsi
  __int64 v29; // rdx
  int v30; // edi
  unsigned __int8 *v31; // r13
  bool v33; // al
  int v34; // eax
  __int64 v35; // rdi
  unsigned __int8 *v36; // r13
  __int64 v37; // r8
  _BYTE *v38; // rax
  unsigned __int8 *v39; // rcx
  bool v40; // dl
  _BYTE *v41; // rbx
  void *v42; // rax
  char v43; // al
  _BYTE *v44; // rbx
  bool v45; // al
  void **v46; // rax
  void **v47; // rbx
  void **v48; // rbx
  bool v49; // al
  bool v50; // al
  unsigned int v51; // ebx
  char *v52; // rax
  char v53; // al
  void *v54; // rax
  bool v55; // al
  unsigned int v56; // ebx
  char *v57; // rax
  char v58; // al
  void *v59; // rax
  void *v60; // rax
  unsigned __int8 v61; // [rsp+Bh] [rbp-145h]
  unsigned int v62; // [rsp+Ch] [rbp-144h]
  bool v63; // [rsp+Ch] [rbp-144h]
  int v64; // [rsp+Ch] [rbp-144h]
  __int64 v65; // [rsp+10h] [rbp-140h]
  __int64 v66; // [rsp+10h] [rbp-140h]
  __int64 v67; // [rsp+10h] [rbp-140h]
  int v68; // [rsp+10h] [rbp-140h]
  __int64 v69; // [rsp+18h] [rbp-138h]
  bool v70; // [rsp+18h] [rbp-138h]
  __int64 v71; // [rsp+18h] [rbp-138h]
  __int64 v72; // [rsp+18h] [rbp-138h]
  unsigned __int8 *v73; // [rsp+18h] [rbp-138h]
  __int64 v74; // [rsp+18h] [rbp-138h]
  unsigned __int8 v75; // [rsp+18h] [rbp-138h]
  __int64 v76; // [rsp+18h] [rbp-138h]
  __int64 v77; // [rsp+20h] [rbp-130h]
  unsigned __int8 *v78; // [rsp+20h] [rbp-130h]
  bool v79; // [rsp+20h] [rbp-130h]
  unsigned __int8 v80; // [rsp+20h] [rbp-130h]
  bool v81; // [rsp+20h] [rbp-130h]
  bool v82; // [rsp+20h] [rbp-130h]
  bool v83; // [rsp+20h] [rbp-130h]
  __int64 v84; // [rsp+20h] [rbp-130h]
  unsigned __int8 v85; // [rsp+20h] [rbp-130h]
  __int64 v86; // [rsp+20h] [rbp-130h]
  __int64 v88; // [rsp+38h] [rbp-118h] BYREF
  __int64 v89; // [rsp+40h] [rbp-110h] BYREF
  __int64 v90; // [rsp+48h] [rbp-108h]
  _QWORD v91[2]; // [rsp+50h] [rbp-100h] BYREF
  char v92[32]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v93; // [rsp+80h] [rbp-D0h]
  _QWORD *v94[2]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE v95[16]; // [rsp+A0h] [rbp-B0h] BYREF
  __int16 v96; // [rsp+B0h] [rbp-A0h]
  __int64 v97; // [rsp+C0h] [rbp-90h]
  __int64 v98; // [rsp+C8h] [rbp-88h]
  __int16 v99; // [rsp+D0h] [rbp-80h]
  __int64 v100; // [rsp+D8h] [rbp-78h]
  void **v101; // [rsp+E0h] [rbp-70h]
  void **v102; // [rsp+E8h] [rbp-68h]
  __int64 v103; // [rsp+F0h] [rbp-60h]
  int v104; // [rsp+F8h] [rbp-58h]
  __int16 v105; // [rsp+FCh] [rbp-54h]
  char v106; // [rsp+FEh] [rbp-52h]
  __int64 v107; // [rsp+100h] [rbp-50h]
  __int64 v108; // [rsp+108h] [rbp-48h]
  void *v109; // [rsp+110h] [rbp-40h] BYREF
  void *v110; // [rsp+118h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v2 > 0x15u )
    return 0;
  v4 = sub_B43CC0(a2);
  v5 = *(_QWORD *)(a2 - 64);
  v6 = v4;
  v94[0] = &v88;
  if ( (unsigned __int8)sub_995E90(v94, v5, v7, v8, v9) )
  {
    v5 = v2;
    v29 = sub_96E680(12, v2);
    if ( v29 )
    {
      v96 = 257;
      v28 = v88;
      v30 = 21;
      goto LABEL_16;
    }
  }
  v10 = sub_B451C0(a2);
  if ( !v10 )
    goto LABEL_10;
  v14 = *(_QWORD *)(a2 - 32);
  v15 = *(_BYTE *)v14;
  if ( *(_BYTE *)v14 == 18 )
  {
    v16 = *(void **)(v14 + 24);
    v77 = *(_QWORD *)(a2 - 32);
    v17 = sub_C33340();
    if ( v16 == v17 )
    {
      v18 = *(_QWORD *)(v77 + 32);
      v5 = *(_BYTE *)(v18 + 20) & 7;
      if ( (_BYTE)v5 != 3 )
      {
        v33 = sub_B451E0(a2);
        v12 = v77;
        if ( !v33 )
          goto LABEL_10;
        goto LABEL_23;
      }
    }
    else
    {
      if ( (*(_BYTE *)(v77 + 44) & 7) != 3 )
      {
        v50 = sub_B451E0(a2);
        v12 = v77;
        if ( !v50 )
          goto LABEL_10;
        goto LABEL_48;
      }
      v18 = v77 + 24;
    }
    if ( (*(_BYTE *)(v18 + 20) & 8) == 0 )
      goto LABEL_25;
    v69 = (__int64)v17;
    if ( !sub_B451E0(a2) )
      goto LABEL_10;
    v12 = v77;
    v11 = v69;
LABEL_63:
    if ( v16 == (void *)v11 )
    {
LABEL_23:
      v12 = *(_QWORD *)(v12 + 32);
      goto LABEL_24;
    }
LABEL_48:
    v12 += 24;
LABEL_24:
    if ( (*(_BYTE *)(v12 + 20) & 7) == 3 )
      goto LABEL_25;
LABEL_10:
    if ( (unsigned __int8)sub_AD8130(v2, v5, v11, v12, v13) || sub_B451F0(a2) && sub_AD7F90(v2, v5, v19, v20, v21) )
    {
      v22 = sub_AD8DD0(*(_QWORD *)(a2 + 8), 1.0);
      v23 = sub_96E6C0(0x15u, (__int64)v22, (_BYTE *)v2, v6);
      v27 = v23;
      if ( v23 )
      {
        if ( sub_AD7F90(v23, (__int64)v22, v24, v25, v26) )
        {
          v28 = *(_QWORD *)(a2 - 64);
          v96 = 257;
          v29 = v27;
          v30 = 18;
LABEL_16:
          v31 = (unsigned __int8 *)sub_B504D0(v30, v28, v29, (__int64)v94, 0, 0);
          sub_B45260(v31, a2, 1);
          return v31;
        }
      }
    }
    return 0;
  }
  v37 = *(_QWORD *)(v14 + 8);
  v5 = (unsigned int)*(unsigned __int8 *)(v37 + 8) - 17;
  if ( (unsigned int)v5 <= 1 )
  {
    v65 = *(_QWORD *)(v14 + 8);
    if ( v15 > 0x15u )
    {
      v74 = *(_QWORD *)(a2 - 32);
      v83 = v10;
      v55 = sub_B451E0(a2);
      v11 = v83;
      v12 = v74;
      v13 = v65;
      if ( !v55 )
        goto LABEL_10;
      goto LABEL_38;
    }
    v5 = 0;
    v78 = *(unsigned __int8 **)(a2 - 32);
    v70 = v10;
    v38 = sub_AD7630(v14, 0, v11);
    v39 = v78;
    v40 = v70;
    v41 = v38;
    if ( v38 && *v38 == 18 )
    {
      v42 = sub_C33340();
      v40 = v70;
      if ( *((void **)v41 + 3) == v42 )
      {
        v44 = (_BYTE *)*((_QWORD *)v41 + 4);
        if ( (v44[20] & 7) == 3 )
        {
LABEL_33:
          if ( (v44[20] & 8) == 0 )
            goto LABEL_25;
        }
      }
      else
      {
        v43 = v41[44];
        v44 = v41 + 24;
        if ( (v43 & 7) == 3 )
          goto LABEL_33;
      }
    }
    else if ( *(_BYTE *)(v65 + 8) == 17 )
    {
      v68 = *(_DWORD *)(v65 + 32);
      if ( v68 )
      {
        v63 = 0;
        v51 = 0;
        while ( 1 )
        {
          v82 = v40;
          v73 = v39;
          v52 = (char *)sub_AD69F0(v39, v51);
          v40 = v82;
          v5 = (unsigned __int64)v52;
          if ( !v52 )
            break;
          v53 = *v52;
          v39 = v73;
          if ( v53 != 13 )
          {
            if ( v53 != 18 )
              break;
            v63 = v82;
            v54 = sub_C33340();
            v40 = v82;
            v39 = v73;
            if ( *(void **)(v5 + 24) == v54 )
            {
              v5 = *(_QWORD *)(v5 + 32);
              if ( (*(_BYTE *)(v5 + 20) & 7) != 3 )
                break;
            }
            else
            {
              if ( (*(_BYTE *)(v5 + 44) & 7) != 3 )
                break;
              v5 += 24LL;
            }
            if ( (*(_BYTE *)(v5 + 20) & 8) != 0 )
              break;
          }
          if ( v68 == ++v51 )
          {
            if ( v63 )
              goto LABEL_25;
            break;
          }
        }
      }
    }
    v79 = v40;
    v45 = sub_B451E0(a2);
    v11 = v79;
    if ( !v45 )
      goto LABEL_10;
    v12 = *(_QWORD *)(a2 - 32);
    v15 = *(_BYTE *)v12;
    if ( *(_BYTE *)v12 == 18 )
    {
      v86 = *(_QWORD *)(a2 - 32);
      v60 = sub_C33340();
      v12 = v86;
      v11 = (__int64)v60;
      v16 = *(void **)(v86 + 24);
      goto LABEL_63;
    }
    v13 = *(_QWORD *)(v12 + 8);
    v5 = (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17;
    goto LABEL_37;
  }
  v62 = *(unsigned __int8 *)(v37 + 8) - 17;
  v67 = *(_QWORD *)(v14 + 8);
  v72 = *(_QWORD *)(a2 - 32);
  v81 = v10;
  v49 = sub_B451E0(a2);
  v11 = v81;
  v12 = v72;
  v13 = v67;
  v5 = v62;
  if ( !v49 )
    goto LABEL_10;
LABEL_37:
  if ( (unsigned int)v5 > 1 )
    goto LABEL_10;
LABEL_38:
  v71 = v13;
  v80 = v11;
  if ( v15 > 0x15u )
    goto LABEL_10;
  v5 = 0;
  v66 = v12;
  v46 = (void **)sub_AD7630(v12, 0, v11);
  v12 = v66;
  v11 = v80;
  v13 = v71;
  v47 = v46;
  if ( !v46 || *(_BYTE *)v46 != 18 )
  {
    if ( *(_BYTE *)(v71 + 8) == 17 )
    {
      v64 = *(_DWORD *)(v71 + 32);
      if ( v64 )
      {
        v61 = 0;
        v56 = 0;
        while ( 1 )
        {
          v75 = v11;
          v84 = v12;
          v57 = (char *)sub_AD69F0((unsigned __int8 *)v12, v56);
          v5 = (unsigned __int64)v57;
          if ( !v57 )
            break;
          v58 = *v57;
          v12 = v84;
          v11 = v75;
          if ( v58 != 13 )
          {
            v76 = v84;
            v85 = v11;
            if ( v58 != 18 )
              goto LABEL_10;
            v59 = sub_C33340();
            v11 = v85;
            v12 = v76;
            if ( *(void **)(v5 + 24) == v59 )
              v5 = *(_QWORD *)(v5 + 32);
            else
              v5 += 24LL;
            if ( (*(_BYTE *)(v5 + 20) & 7) != 3 )
              goto LABEL_10;
            v61 = v85;
          }
          if ( v64 == ++v56 )
          {
            if ( v61 )
              goto LABEL_25;
            goto LABEL_10;
          }
        }
      }
    }
    goto LABEL_10;
  }
  if ( v46[3] == sub_C33340() )
    v48 = (void **)v47[4];
  else
    v48 = v47 + 3;
  if ( (*((_BYTE *)v48 + 20) & 7) != 3 )
    goto LABEL_10;
LABEL_25:
  v100 = sub_BD5C60(a2);
  v94[1] = (_QWORD *)0x200000000LL;
  v102 = &v110;
  v109 = &unk_49DA100;
  v105 = 512;
  v99 = 0;
  v110 = &unk_49DA0B0;
  v94[0] = v95;
  v101 = &v109;
  v103 = 0;
  v104 = 0;
  v106 = 7;
  v107 = 0;
  v108 = 0;
  v97 = 0;
  v98 = 0;
  sub_D5F1F0((__int64)v94, a2);
  v93 = 257;
  v34 = sub_B45210(a2);
  v35 = *(_QWORD *)(a2 + 8);
  BYTE4(v90) = 1;
  LODWORD(v90) = v34;
  v91[0] = sub_AD9500(v35, 0);
  v91[1] = *(_QWORD *)(a2 - 64);
  v89 = *(_QWORD *)(v2 + 8);
  v36 = (unsigned __int8 *)sub_B33D10((__int64)v94, 0x1Au, (__int64)&v89, 1, (int)v91, 2, v90, (__int64)v92);
  sub_BD6B90(v36, (unsigned __int8 *)a2);
  v31 = sub_F162A0(a1, a2, (__int64)v36);
  nullsub_61();
  v109 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v94[0] != v95 )
    _libc_free(v94[0], a2);
  return v31;
}
