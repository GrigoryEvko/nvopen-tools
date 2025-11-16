// Function: sub_3725050
// Address: 0x3725050
//
void __fastcall sub_3725050(
        const void *a1,
        __int64 a2,
        _QWORD *a3,
        char *a4,
        __int64 a5,
        __int64 a6,
        void *src,
        __int64 a8)
{
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  signed __int64 v12; // r12
  __int64 v13; // r12
  __int64 v14; // rdi
  void (*v15)(); // rax
  __int64 v16; // rdi
  void (*v17)(); // rax
  __int64 v18; // rdi
  void (*v19)(); // rax
  __int64 v20; // rdi
  void (*v21)(); // rax
  __int64 v22; // rdi
  void (*v23)(); // rax
  __int64 v24; // rdi
  void (*v25)(); // rax
  __int64 v26; // r13
  __int64 v27; // rdi
  void (*v28)(); // rax
  __int64 v29; // rdi
  void (*v30)(); // rax
  unsigned __int16 *v31; // r12
  __int64 v32; // r15
  void (*v33)(); // r14
  const char *v34; // rax
  __int64 v35; // rdx
  int v36; // esi
  __int64 v37; // r15
  void (*v38)(); // r14
  const char *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r13
  unsigned __int64 v42; // r12
  int v43; // r15d
  __int64 v44; // rdi
  __int64 v45; // r9
  void (*v46)(); // rax
  __int64 *v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rdx
  __int64 v51; // rsi
  __int64 ***v52; // r14
  __int64 **v53; // r15
  __int64 **v54; // r12
  __int64 v55; // rdi
  __int64 v56; // r8
  void (*v57)(); // rcx
  __int64 v58; // rdx
  __int64 *v59; // rdx
  __int64 v60; // rsi
  __int64 v61; // r9
  __int64 *v62; // rdx
  __int64 v63; // rsi
  __int64 v64; // rdi
  __int64 v65; // r8
  void (*v66)(); // rcx
  __int64 *v67; // rdx
  _QWORD *v68; // r8
  _QWORD *v69; // rcx
  int v70; // edx
  unsigned __int64 v71; // rdx
  unsigned __int16 *v72; // rdi
  __int64 ***i; // [rsp+10h] [rbp-F0h]
  _QWORD *v74; // [rsp+20h] [rbp-E0h]
  unsigned __int16 *v75; // [rsp+28h] [rbp-D8h]
  _QWORD *v76; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v77; // [rsp+38h] [rbp-C8h] BYREF
  char *v78; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v79; // [rsp+48h] [rbp-B8h]
  unsigned __int64 *v80; // [rsp+50h] [rbp-B0h]
  __int16 v81; // [rsp+60h] [rbp-A0h]
  _QWORD *v82; // [rsp+70h] [rbp-90h] BYREF
  __int64 v83; // [rsp+78h] [rbp-88h]
  char v84; // [rsp+80h] [rbp-80h]
  __int64 v85; // [rsp+84h] [rbp-7Ch]
  __int64 v86; // [rsp+8Ch] [rbp-74h]
  int v87; // [rsp+94h] [rbp-6Ch]
  int v88; // [rsp+98h] [rbp-68h]
  unsigned __int16 *v89; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v90; // [rsp+A8h] [rbp-58h]
  _BYTE v91[16]; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v92; // [rsp+C0h] [rbp-40h]

  sub_3724C70(a2, a1, a3, a4, a5, a6);
  v11 = *(_QWORD *)(a2 + 152);
  v83 = a2;
  v82 = a1;
  v86 = v11;
  v12 = 4 * a8;
  v87 = 4 * a8 + 8;
  v89 = (unsigned __int16 *)v91;
  v84 = 1;
  v85 = 0x148415348LL;
  v88 = 0;
  v90 = 0x400000000LL;
  if ( (unsigned __int64)(4 * a8) > 0x10 )
  {
    sub_C8D5F0((__int64)&v89, v91, v12 >> 2, 4u, v9, v10);
    v72 = &v89[2 * (unsigned int)v90];
  }
  else
  {
    if ( !v12 )
      goto LABEL_3;
    v72 = (unsigned __int16 *)v91;
  }
  memcpy(v72, src, v12);
  LODWORD(v12) = v90;
LABEL_3:
  v92 = a5;
  LODWORD(v90) = ((4 * a8) >> 2) + v12;
  v13 = (__int64)v82;
  v14 = v82[28];
  v15 = *(void (**)())(*(_QWORD *)v14 + 120LL);
  v78 = "Header Magic";
  v81 = 259;
  if ( v15 != nullsub_98 )
    ((void (__fastcall *)(__int64, char **, __int64))v15)(v14, &v78, 1);
  sub_31DCA10(v13, v85);
  v16 = *(_QWORD *)(v13 + 224);
  v17 = *(void (**)())(*(_QWORD *)v16 + 120LL);
  v78 = "Header Version";
  v81 = 259;
  if ( v17 != nullsub_98 )
    ((void (__fastcall *)(__int64, char **, __int64))v17)(v16, &v78, 1);
  sub_31DC9F0(v13, WORD2(v85));
  v18 = *(_QWORD *)(v13 + 224);
  v19 = *(void (**)())(*(_QWORD *)v18 + 120LL);
  v78 = "Header Hash Function";
  v81 = 259;
  if ( v19 != nullsub_98 )
    ((void (__fastcall *)(__int64, char **, __int64))v19)(v18, &v78, 1);
  sub_31DC9F0(v13, HIWORD(v85));
  v20 = *(_QWORD *)(v13 + 224);
  v21 = *(void (**)())(*(_QWORD *)v20 + 120LL);
  v78 = "Header Bucket Count";
  v81 = 259;
  if ( v21 != nullsub_98 )
    ((void (__fastcall *)(__int64, char **, __int64))v21)(v20, &v78, 1);
  sub_31DCA10(v13, v86);
  v22 = *(_QWORD *)(v13 + 224);
  v23 = *(void (**)())(*(_QWORD *)v22 + 120LL);
  v78 = "Header Hash Count";
  v81 = 259;
  if ( v23 != nullsub_98 )
    ((void (__fastcall *)(__int64, char **, __int64))v23)(v22, &v78, 1);
  sub_31DCA10(v13, SHIDWORD(v86));
  v24 = *(_QWORD *)(v13 + 224);
  v25 = *(void (**)())(*(_QWORD *)v24 + 120LL);
  v78 = "Header Data Length";
  v81 = 259;
  if ( v25 != nullsub_98 )
    ((void (__fastcall *)(__int64, char **, __int64))v25)(v24, &v78, 1);
  sub_31DCA10(v13, v87);
  v26 = (__int64)v82;
  v27 = v82[28];
  v28 = *(void (**)())(*(_QWORD *)v27 + 120LL);
  v78 = "HeaderData Die Offset Base";
  v81 = 259;
  if ( v28 != nullsub_98 )
    ((void (__fastcall *)(__int64, char **, __int64))v28)(v27, &v78, 1);
  sub_31DCA10(v26, v88);
  v29 = *(_QWORD *)(v26 + 224);
  v30 = *(void (**)())(*(_QWORD *)v29 + 120LL);
  v78 = "HeaderData Atom Count";
  v81 = 259;
  if ( v30 != nullsub_98 )
    ((void (__fastcall *)(__int64, char **, __int64))v30)(v29, &v78, 1);
  sub_31DCA10(v26, v90);
  v31 = v89;
  v75 = &v89[2 * (unsigned int)v90];
  if ( v89 != v75 )
  {
    do
    {
      v37 = *(_QWORD *)(v26 + 224);
      v38 = *(void (**)())(*(_QWORD *)v37 + 120LL);
      v39 = sub_E0CA30(*v31);
      v81 = 261;
      v78 = (char *)v39;
      v79 = v40;
      if ( v38 != nullsub_98 )
        ((void (__fastcall *)(__int64, char **, __int64))v38)(v37, &v78, 1);
      sub_31DC9F0(v26, *v31);
      v32 = *(_QWORD *)(v26 + 224);
      v33 = *(void (**)())(*(_QWORD *)v32 + 120LL);
      v34 = sub_E06AB0(v31[1]);
      v81 = 261;
      v78 = (char *)v34;
      v79 = v35;
      if ( v33 != nullsub_98 )
        ((void (__fastcall *)(__int64, char **, __int64))v33)(v32, &v78, 1);
      v36 = v31[1];
      v31 += 2;
      sub_31DC9F0(v26, v36);
    }
    while ( v75 != v31 );
  }
  v77 = 0;
  v41 = *(_QWORD *)(v83 + 184);
  v42 = 0xAAAAAAAAAAAAAAABLL * ((*(_QWORD *)(v83 + 192) - v41) >> 3);
  if ( v42 )
  {
    v43 = 0;
    do
    {
      v44 = (__int64)v82;
      v45 = v82[28];
      v46 = *(void (**)())(*(_QWORD *)v45 + 120LL);
      v78 = "Bucket ";
      v80 = &v77;
      v81 = 2819;
      if ( v46 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, char **, __int64))v46)(v45, &v78, 1);
        v44 = (__int64)v82;
      }
      if ( *(_QWORD *)(v41 + 24 * v77 + 8) == *(_QWORD *)(v41 + 24 * v77) )
        sub_31DCA10(v44, -1);
      else
        sub_31DCA10(v44, v43);
      v47 = (__int64 *)(v41 + 24 * v77);
      v48 = *v47;
      v49 = v47[1];
      if ( *v47 != v49 )
      {
        v50 = -1;
        do
        {
          v51 = v50;
          v50 = *(unsigned int *)(*(_QWORD *)v48 + 8LL);
          v48 += 8;
          v43 += v51 != v50;
        }
        while ( v49 != v48 );
      }
      ++v77;
    }
    while ( v42 > v77 );
  }
  sub_3722350((__int64)&v82);
  sub_3722650((__int64)&v82);
  v52 = *(__int64 ****)(v83 + 184);
  for ( i = *(__int64 ****)(v83 + 192); i != v52; v52 += 3 )
  {
    v53 = *v52;
    v54 = v52[1];
    if ( *v52 != v54 )
    {
      while ( 1 )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(*(_QWORD *)v82[28] + 208LL))(v82[28], (*v53)[5], 0);
        v55 = (__int64)v82;
        v56 = v82[28];
        v57 = *(void (**)())(*(_QWORD *)v56 + 120LL);
        v58 = **v53;
        if ( (v58 & 4) != 0 )
        {
          v71 = v58 & 0xFFFFFFFFFFFFFFF8LL;
          v61 = *(_QWORD *)(v71 + 24);
          v60 = *(_QWORD *)(v71 + 32);
        }
        else
        {
          v59 = (__int64 *)(v58 & 0xFFFFFFFFFFFFFFF8LL);
          v60 = *v59;
          v61 = (__int64)(v59 + 4);
        }
        v78 = (char *)v61;
        v81 = 261;
        v79 = v60;
        if ( v57 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, char **, __int64))v57)(v56, &v78, 1);
          v55 = (__int64)v82;
        }
        v62 = (__int64 *)(**v53 & 0xFFFFFFFFFFFFFFF8LL);
        v63 = (__int64)(v62 + 1);
        if ( (**v53 & 4) == 0 )
          ++v62;
        sub_31F0E70(v55, v63, (__int64)v62, **v53 & 4, v56, v61, *v62, v62[1]);
        v64 = (__int64)v82;
        v65 = v82[28];
        v66 = *(void (**)())(*(_QWORD *)v65 + 120LL);
        v78 = "Num DIEs";
        v81 = 259;
        if ( v66 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, char **, __int64))v66)(v65, &v78, 1);
          v64 = (__int64)v82;
        }
        sub_31DCA10(v64, ((*v53)[3] - (*v53)[2]) >> 3);
        v67 = *v53;
        v68 = (_QWORD *)(*v53)[3];
        v69 = (_QWORD *)(*v53)[2];
        if ( v68 != v69 )
        {
          do
          {
            v74 = v68;
            v76 = v69;
            (*(void (__fastcall **)(_QWORD, _QWORD *))(*(_QWORD *)*v69 + 24LL))(*v69, v82);
            v68 = v74;
            v69 = v76 + 1;
          }
          while ( v74 != v76 + 1 );
          v67 = *v53;
        }
        ++v53;
        v70 = *((_DWORD *)v67 + 2);
        if ( v54 == v53 )
          break;
        if ( *((_DWORD *)*v53 + 2) != v70 )
          sub_31DCA10((__int64)v82, 0);
      }
      if ( v52[1] != *v52 )
        sub_31DCA10((__int64)v82, 0);
    }
  }
  if ( v89 != (unsigned __int16 *)v91 )
    _libc_free((unsigned __int64)v89);
}
