// Function: sub_23D4F90
// Address: 0x23d4f90
//
__int64 __fastcall sub_23D4F90(__int64 a1, __int64 a2)
{
  unsigned int v3; // r13d
  int v6; // eax
  __int64 *v7; // rax
  unsigned __int8 *v8; // rbx
  unsigned int v9; // eax
  unsigned int v10; // r10d
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  unsigned __int64 v16; // rax
  int v17; // edx
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // r15
  _BYTE *v21; // rbx
  unsigned __int64 v22; // rax
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rax
  __int16 v26; // dx
  __int64 v27; // rbx
  __int64 v28; // rax
  unsigned int v29; // r10d
  __int64 v30; // rdi
  __int64 *v31; // rax
  __int64 v32; // rsi
  __int64 v33; // r9
  __int64 v34; // r8
  __int64 v35; // rsi
  unsigned __int64 v36; // rcx
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rax
  unsigned __int8 *v39; // rdi
  unsigned __int8 *v40; // rax
  char v41; // al
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned int v44; // eax
  char v45; // al
  unsigned __int8 *v46; // rax
  _QWORD *v47; // rcx
  __int64 v48; // [rsp+8h] [rbp-178h]
  char v49; // [rsp+18h] [rbp-168h]
  unsigned int v50; // [rsp+18h] [rbp-168h]
  unsigned __int8 *v51; // [rsp+20h] [rbp-160h]
  char v52; // [rsp+20h] [rbp-160h]
  unsigned int v53; // [rsp+20h] [rbp-160h]
  unsigned int v54; // [rsp+20h] [rbp-160h]
  unsigned int v55; // [rsp+28h] [rbp-158h]
  unsigned __int8 *v56; // [rsp+48h] [rbp-138h] BYREF
  unsigned __int8 *v57; // [rsp+50h] [rbp-130h] BYREF
  __int64 v58; // [rsp+58h] [rbp-128h] BYREF
  __int64 v59; // [rsp+60h] [rbp-120h] BYREF
  unsigned int v60; // [rsp+68h] [rbp-118h]
  int v61; // [rsp+6Ch] [rbp-114h]
  _QWORD v62[4]; // [rsp+70h] [rbp-110h] BYREF
  __int64 v63; // [rsp+90h] [rbp-F0h] BYREF
  int v64; // [rsp+98h] [rbp-E8h]
  __int16 v65; // [rsp+B0h] [rbp-D0h]
  unsigned __int64 v66; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+C8h] [rbp-B8h]
  __int64 *v68; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v69; // [rsp+D8h] [rbp-A8h]
  __int64 v70; // [rsp+E0h] [rbp-A0h]
  __int64 v71; // [rsp+F0h] [rbp-90h]
  __int64 v72; // [rsp+F8h] [rbp-88h]
  __int16 v73; // [rsp+100h] [rbp-80h]
  __int64 v74; // [rsp+108h] [rbp-78h]
  void **v75; // [rsp+110h] [rbp-70h]
  void **v76; // [rsp+118h] [rbp-68h]
  __int64 v77; // [rsp+120h] [rbp-60h]
  int v78; // [rsp+128h] [rbp-58h]
  __int16 v79; // [rsp+12Ch] [rbp-54h]
  char v80; // [rsp+12Eh] [rbp-52h]
  __int64 v81; // [rsp+130h] [rbp-50h]
  __int64 v82; // [rsp+138h] [rbp-48h]
  void *v83; // [rsp+140h] [rbp-40h] BYREF
  void *v84; // [rsp+148h] [rbp-38h] BYREF

  if ( *(_BYTE *)a1 != 84 )
    return 0;
  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 2 )
    return 0;
  v6 = sub_BCB060(*(_QWORD *)(a1 + 8));
  if ( !v6 || (v6 & (v6 - 1)) != 0 )
    return 0;
  v7 = *(__int64 **)(a1 - 8);
  v8 = (unsigned __int8 *)*v7;
  v51 = (unsigned __int8 *)v7[4];
  v9 = sub_23D4B90(*v7, &v56, &v57, &v58);
  v10 = v9;
  if ( !v9 )
  {
LABEL_46:
    v44 = sub_23D4B90((__int64)v51, &v56, &v57, &v58);
    v10 = v44;
    if ( v44 )
    {
      if ( v44 == 180 )
      {
        v11 = 0;
        v12 = 8;
        if ( v56 == v8 )
          goto LABEL_10;
      }
      else
      {
        v11 = 0;
        v12 = 8;
        if ( v44 != 181 || v57 == v8 )
          goto LABEL_10;
      }
    }
    return 0;
  }
  if ( v9 == 180 )
  {
    v11 = 8;
    v12 = 0;
    if ( v56 == v51 )
      goto LABEL_10;
    goto LABEL_46;
  }
  v11 = 8;
  v12 = 0;
  if ( v9 == 181 && v57 != v51 )
    goto LABEL_46;
LABEL_10:
  v13 = *(_QWORD *)(a1 - 8) + 32LL * *(unsigned int *)(a1 + 72);
  v14 = *(_QWORD *)(v13 + v11);
  v15 = *(_QWORD *)(v13 + v12);
  v16 = *(_QWORD *)(v14 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v16 == v14 + 48 )
  {
    v18 = 0;
  }
  else
  {
    if ( !v16 )
      BUG();
    v17 = *(unsigned __int8 *)(v16 - 24);
    v18 = 0;
    v19 = v16 - 24;
    if ( (unsigned int)(v17 - 30) < 0xB )
      v18 = v19;
  }
  v55 = v10;
  if ( !(unsigned __int8)sub_B19DB0(a2, (__int64)v56, v18) )
    return 0;
  if ( !(unsigned __int8)sub_B19DB0(a2, (__int64)v57, v18) )
    return 0;
  v20 = *(_QWORD *)(a1 + 40);
  v66 = 32;
  v68 = 0;
  v67 = v58;
  v69 = v20;
  v70 = v15;
  if ( *(_BYTE *)v18 != 31 )
    return 0;
  if ( (*(_DWORD *)(v18 + 4) & 0x7FFFFFF) != 3 )
    return 0;
  v21 = *(_BYTE **)(v18 - 96);
  if ( *v21 != 82 )
    return 0;
  v22 = sub_B53900(*(_QWORD *)(v18 - 96));
  v63 = sub_B53630(v22, v66);
  v64 = v23;
  if ( !(_BYTE)v23 )
    return 0;
  if ( *((_QWORD *)v21 - 8) != v67 )
    return 0;
  v3 = sub_10081F0(&v68, *((_QWORD *)v21 - 4));
  if ( !(_BYTE)v3 )
    return 0;
  v24 = *(_QWORD *)(v18 - 32);
  if ( !v24 )
    return 0;
  if ( v24 != v69 )
    return 0;
  v25 = *(_QWORD *)(v18 - 64);
  if ( !v25 || v25 != v70 )
    return 0;
  v27 = sub_AA5190(v20);
  if ( v27 )
  {
    v52 = v26;
    v49 = HIBYTE(v26);
  }
  else
  {
    v49 = 0;
    v52 = 0;
  }
  v28 = sub_AA48A0(v20);
  v75 = &v83;
  v74 = v28;
  v29 = v55;
  v76 = &v84;
  v66 = (unsigned __int64)&v68;
  v67 = 0x200000000LL;
  v83 = &unk_49DA100;
  v77 = 0;
  v78 = 0;
  v84 = &unk_49DA0B0;
  LOBYTE(v28) = v52;
  v79 = 512;
  BYTE1(v28) = v49;
  v80 = 7;
  v81 = 0;
  v82 = 0;
  v71 = v20;
  v72 = v27;
  v73 = v28;
  if ( v27 != v20 + 48 )
  {
    v30 = v27 - 24;
    if ( !v27 )
      v30 = 0;
    v31 = (__int64 *)sub_B46C60(v30);
    v29 = v55;
    v32 = *v31;
    v63 = *v31;
    if ( v63 && (sub_B96E90((__int64)&v63, v32, 1), v34 = v63, v29 = v55, v63) )
    {
      v35 = (unsigned int)v67;
      v36 = v66;
      v37 = v66 + 16LL * (unsigned int)v67;
      if ( v66 != v37 )
      {
        v38 = v66;
        while ( *(_DWORD *)v38 )
        {
          v38 += 16LL;
          if ( v37 == v38 )
            goto LABEL_56;
        }
        *(_QWORD *)(v38 + 8) = v63;
        goto LABEL_38;
      }
LABEL_56:
      if ( (unsigned int)v67 >= (unsigned __int64)HIDWORD(v67) )
      {
        if ( HIDWORD(v67) < (unsigned __int64)(unsigned int)v67 + 1 )
        {
          v48 = v63;
          sub_C8D5F0((__int64)&v66, &v68, (unsigned int)v67 + 1LL, 0x10u, v63, v33);
          v36 = v66;
          v35 = (unsigned int)v67;
          v34 = v48;
          v29 = v55;
        }
        v47 = (_QWORD *)(16 * v35 + v36);
        *v47 = 0;
        v47[1] = v34;
        v34 = v63;
        LODWORD(v67) = v67 + 1;
      }
      else
      {
        if ( v37 )
        {
          *(_DWORD *)v37 = 0;
          *(_QWORD *)(v37 + 8) = v34;
          v34 = v63;
        }
        LODWORD(v67) = v67 + 1;
      }
    }
    else
    {
      v50 = v29;
      sub_93FB40((__int64)&v66, 0);
      v34 = v63;
      v29 = v50;
    }
    if ( v34 )
    {
LABEL_38:
      v53 = v29;
      sub_B91220((__int64)&v63, v34);
      v29 = v53;
    }
  }
  v39 = v57;
  v40 = v57;
  if ( v56 == v57 )
    goto LABEL_43;
  v54 = v29;
  if ( v29 == 180 )
  {
    v45 = sub_98ED70(v57, 0, 0, 0, 0);
    v29 = 180;
    if ( v45 )
      goto LABEL_42;
    v65 = 257;
    v40 = (unsigned __int8 *)sub_1156690((__int64 *)&v66, (__int64)v57, (__int64)&v63);
    v39 = v56;
    v57 = v40;
    v29 = 180;
  }
  else
  {
    v41 = sub_98ED70(v56, 0, 0, 0, 0);
    v29 = v54;
    if ( v41 )
    {
LABEL_42:
      v39 = v56;
      v40 = v57;
      goto LABEL_43;
    }
    v65 = 257;
    v46 = (unsigned __int8 *)sub_1156690((__int64 *)&v66, (__int64)v56, (__int64)&v63);
    v29 = v54;
    v56 = v46;
    v39 = v46;
    v40 = v57;
  }
LABEL_43:
  v62[1] = v40;
  v61 = 0;
  v62[2] = v58;
  v42 = *(_QWORD *)(a1 + 8);
  v65 = 257;
  v62[0] = v39;
  v59 = v42;
  v43 = sub_B33D10((__int64)&v66, v29, (__int64)&v59, 1, (int)v62, 3, v60, (__int64)&v63);
  sub_BD84D0(a1, v43);
  nullsub_61();
  v83 = &unk_49DA100;
  nullsub_63();
  if ( (__int64 **)v66 != &v68 )
    _libc_free(v66);
  return v3;
}
