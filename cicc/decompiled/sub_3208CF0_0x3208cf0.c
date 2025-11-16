// Function: sub_3208CF0
// Address: 0x3208cf0
//
void __fastcall sub_3208CF0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rbx
  _BYTE *v4; // rdx
  _BYTE *v5; // r13
  _BYTE *v6; // rax
  unsigned __int8 v7; // dl
  __int64 v8; // rax
  char v9; // al
  char *v10; // rdx
  _BYTE *v11; // r8
  size_t v12; // r14
  char *v13; // rax
  _QWORD *v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r14
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r15
  void (*v25)(); // rax
  unsigned __int8 v26; // al
  _BYTE *v27; // rdx
  __int64 v28; // rdi
  void (*v29)(); // rax
  int v30; // ecx
  __int64 v31; // rsi
  int v32; // ecx
  unsigned int v33; // edx
  __int64 *v34; // rax
  __int64 v35; // r10
  __int64 v36; // rdx
  __int64 v37; // rdi
  void (*v38)(); // rax
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 *v41; // rdi
  void (*v42)(); // rax
  _BYTE *v43; // rax
  char *v44; // rdx
  __int64 v45; // r9
  unsigned __int8 v46; // al
  __int64 v47; // r14
  char v48; // al
  unsigned __int64 v49; // rax
  __int64 v50; // rcx
  unsigned __int8 v51; // al
  __int64 v52; // r14
  unsigned __int8 v53; // al
  char v54; // al
  _QWORD *v55; // rax
  _QWORD *v56; // rdi
  char *v57; // rdx
  unsigned __int8 v58; // al
  unsigned __int8 **v59; // rbx
  int v60; // eax
  int v61; // r8d
  _BYTE *src; // [rsp+0h] [rbp-90h]
  _BYTE *v63; // [rsp+8h] [rbp-88h]
  __int64 v64; // [rsp+8h] [rbp-88h]
  _QWORD *v65; // [rsp+10h] [rbp-80h] BYREF
  char *v66; // [rsp+18h] [rbp-78h]
  _QWORD v67[2]; // [rsp+20h] [rbp-70h] BYREF
  char *v68; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v69; // [rsp+38h] [rbp-58h]
  char v70; // [rsp+3Ch] [rbp-54h]
  char v71; // [rsp+50h] [rbp-40h]
  char v72; // [rsp+51h] [rbp-3Fh]

  v3 = *a2;
  v63 = (_BYTE *)(*a2 - 16);
  if ( (*v63 & 2) != 0 )
    v4 = *(_BYTE **)(v3 - 32);
  else
    v4 = &v63[-8 * ((*v63 >> 2) & 0xF)];
  v5 = *(_BYTE **)v4;
  v6 = (_BYTE *)*((_QWORD *)v4 + 6);
  if ( !v6 || *v6 != 13 )
  {
    if ( *(_BYTE *)(a1 + 800) == 2 )
      goto LABEL_10;
    if ( v5 )
    {
LABEL_9:
      if ( (unsigned __int8)(*v5 - 18) <= 2u )
        goto LABEL_10;
LABEL_40:
      v43 = (_BYTE *)sub_A547D0(v3, 1);
      sub_3205680((__int64)&v65, a1, (__int64)v5, v43, v44, v45);
      v15 = a2[1];
      v16 = v15 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v15 )
        goto LABEL_41;
      goto LABEL_16;
    }
LABEL_39:
    v5 = 0;
    goto LABEL_40;
  }
  v7 = *(v6 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *((_QWORD *)v6 - 4);
  else
    v8 = (__int64)&v6[-8 * ((v7 >> 2) & 0xF) - 16];
  v5 = *(_BYTE **)(v8 + 8);
  v9 = *(_BYTE *)(a1 + 800);
  if ( !v5 )
  {
    if ( v9 == 2 )
      goto LABEL_10;
    goto LABEL_39;
  }
  if ( v9 != 2 )
    goto LABEL_9;
LABEL_10:
  v65 = v67;
  v11 = (_BYTE *)sub_A547D0(v3, 1);
  v12 = (size_t)v10;
  v13 = v10;
  if ( &v11[(_QWORD)v10] && !v11 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v68 = v10;
  if ( (unsigned __int64)v10 > 0xF )
  {
    src = v11;
    v55 = (_QWORD *)sub_22409D0((__int64)&v65, (unsigned __int64 *)&v68, 0);
    v11 = src;
    v65 = v55;
    v56 = v55;
    v67[0] = v68;
LABEL_53:
    memcpy(v56, v11, v12);
    v13 = v68;
    v14 = v65;
    goto LABEL_15;
  }
  if ( v10 == (char *)1 )
  {
    LOBYTE(v67[0]) = *v11;
    v14 = v67;
    goto LABEL_15;
  }
  if ( v10 )
  {
    v56 = v67;
    goto LABEL_53;
  }
  v14 = v67;
LABEL_15:
  v66 = v13;
  v13[(_QWORD)v14] = 0;
  v15 = a2[1];
  v16 = v15 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v15 )
    goto LABEL_41;
LABEL_16:
  if ( (v15 & 4) == 0 && v16 )
  {
    v20 = sub_31DB510(*(_QWORD *)(a1 + 8), v16);
    if ( (*(_BYTE *)(v16 + 33) & 0x1C) != 0 )
      v21 = (unsigned int)(*(_BYTE *)(v3 + 20) == 0) + 4370;
    else
      v21 = (unsigned int)(*(_BYTE *)(v3 + 20) == 0) + 4364;
    v22 = sub_31F8790(a1, v21, v17, v18, v19);
    v23 = *(_QWORD *)(a1 + 528);
    v24 = v22;
    v25 = *(void (**)())(*(_QWORD *)v23 + 120LL);
    v72 = 1;
    v68 = "Type";
    v71 = 3;
    if ( v25 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64, char **, __int64))v25)(v23, &v68, 1);
      v23 = *(_QWORD *)(a1 + 528);
    }
    v26 = *(_BYTE *)(v3 - 16);
    if ( (v26 & 2) != 0 )
      v27 = *(_BYTE **)(v3 - 32);
    else
      v27 = &v63[-8 * ((v26 >> 2) & 0xF)];
    v64 = v23;
    LODWORD(v68) = sub_32053D0(a1, *((_QWORD *)v27 + 3));
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v64 + 536LL))(v64, (unsigned int)v68, 4);
    v28 = *(_QWORD *)(a1 + 528);
    v29 = *(void (**)())(*(_QWORD *)v28 + 120LL);
    v72 = 1;
    v68 = "DataOffset";
    v71 = 3;
    if ( v29 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64, char **, __int64))v29)(v28, &v68, 1);
      v28 = *(_QWORD *)(a1 + 528);
    }
    v30 = *(_DWORD *)(a1 + 832);
    v31 = *(_QWORD *)(a1 + 816);
    if ( v30 )
    {
      v32 = v30 - 1;
      v33 = v32 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v34 = (__int64 *)(v31 + 16LL * v33);
      v35 = *v34;
      if ( v3 == *v34 )
      {
LABEL_28:
        v36 = v34[1];
LABEL_29:
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v28 + 368LL))(v28, v20, v36);
        v37 = *(_QWORD *)(a1 + 528);
        v38 = *(void (**)())(*(_QWORD *)v37 + 120LL);
        v72 = 1;
        v68 = "Segment";
        v71 = 3;
        if ( v38 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, char **, __int64))v38)(v37, &v68, 1);
          v37 = *(_QWORD *)(a1 + 528);
        }
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v37 + 360LL))(v37, v20);
        v41 = *(__int64 **)(a1 + 528);
        v42 = *(void (**)())(*v41 + 120);
        v72 = 1;
        v68 = "Name";
        v71 = 3;
        if ( v42 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64 *, char **, __int64))v42)(v41, &v68, 1);
          v41 = *(__int64 **)(a1 + 528);
        }
        sub_31F4F00(v41, v65, (unsigned __int64)v66, 12, v39, v40);
        sub_31F8930(a1, v24);
        goto LABEL_34;
      }
      v60 = 1;
      while ( v35 != -4096 )
      {
        v61 = v60 + 1;
        v33 = v32 & (v60 + v33);
        v34 = (__int64 *)(v31 + 16LL * v33);
        v35 = *v34;
        if ( v3 == *v34 )
          goto LABEL_28;
        v60 = v61;
      }
    }
    v36 = 0;
    goto LABEL_29;
  }
LABEL_41:
  v46 = *(_BYTE *)(v3 - 16);
  if ( (v46 & 2) != 0 )
  {
    v47 = *(_QWORD *)(*(_QWORD *)(v3 - 32) + 24LL);
    v48 = *(_BYTE *)v47;
    if ( *(_BYTE *)v47 != 14 )
      goto LABEL_43;
  }
  else
  {
    v47 = *(_QWORD *)&v63[-8 * ((v46 >> 2) & 0xF) + 24];
    v48 = *(_BYTE *)v47;
    if ( *(_BYTE *)v47 != 14 )
    {
LABEL_43:
      while ( v48 == 13 )
      {
        v49 = (unsigned int)sub_AF18C0(v47) - 15;
        if ( (unsigned __int16)v49 > 0x33u || (v50 = 0x8000000010003LL, !_bittest64(&v50, v49)) )
        {
          v51 = *(_BYTE *)(v47 - 16);
          v52 = (v51 & 2) != 0 ? *(_QWORD *)(v47 - 32) : v47 - 16 - 8LL * ((v51 >> 2) & 0xF);
          v47 = *(_QWORD *)(v52 + 24);
          v48 = *(_BYTE *)v47;
          if ( *(_BYTE *)v47 != 14 )
            continue;
        }
        goto LABEL_49;
      }
      v54 = 1;
      if ( *(_DWORD *)(v47 + 44) == 4 )
        goto LABEL_55;
LABEL_49:
      v53 = *(_BYTE *)(v3 - 16);
      if ( (v53 & 2) != 0 )
      {
        v47 = *(_QWORD *)(*(_QWORD *)(v3 - 32) + 24LL);
        goto LABEL_51;
      }
      v54 = sub_32120E0(*(_QWORD *)&v63[-8 * ((v53 >> 2) & 0xF) + 24]);
      goto LABEL_55;
    }
  }
LABEL_51:
  v54 = sub_32120E0(v47);
LABEL_55:
  v57 = *(char **)(*(_QWORD *)(v16 + 16) + 8LL);
  v69 = 64;
  v70 = v54;
  v68 = v57;
  v58 = *(_BYTE *)(v3 - 16);
  if ( (v58 & 2) != 0 )
    v59 = *(unsigned __int8 ***)(v3 - 32);
  else
    v59 = (unsigned __int8 **)&v63[-8 * ((v58 >> 2) & 0xF)];
  sub_32086D0(a1, v59[3], (__int64)&v68, (__int64)&v65, (__int64)v11);
  if ( v69 > 0x40 && v68 )
    j_j___libc_free_0_0((unsigned __int64)v68);
LABEL_34:
  if ( v65 != v67 )
    j_j___libc_free_0((unsigned __int64)v65);
}
