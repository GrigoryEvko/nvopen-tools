// Function: sub_39BD6B0
// Address: 0x39bd6b0
//
void __fastcall sub_39BD6B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6, void *src, __int64 a8)
{
  int v9; // r8d
  int v10; // r9d
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
  const char *v33; // rax
  __int64 v34; // rdx
  int v35; // esi
  __int64 v36; // r15
  const char *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r13
  __int64 v40; // r12
  unsigned __int64 v41; // r12
  int v42; // r15d
  __int64 v43; // rdi
  __int64 v44; // r9
  void (*v45)(); // rax
  __int64 *v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 ***v51; // r15
  __int64 ***v52; // r12
  __int64 v53; // rdi
  __int64 v54; // r8
  void (*v55)(); // rcx
  __int64 *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rdi
  __int64 v59; // r8
  void (*v60)(); // rax
  __int64 **v61; // rax
  __int64 *v62; // rdx
  __int64 *v63; // rcx
  int v64; // eax
  unsigned __int16 *v65; // rdi
  unsigned __int64 v66; // [rsp+10h] [rbp-F0h]
  __int64 ****v67; // [rsp+18h] [rbp-E8h]
  __int64 i; // [rsp+20h] [rbp-E0h]
  unsigned __int16 *v69; // [rsp+30h] [rbp-D0h]
  __int64 *v70; // [rsp+30h] [rbp-D0h]
  void (*v71)(); // [rsp+38h] [rbp-C8h]
  void (*v72)(); // [rsp+38h] [rbp-C8h]
  __int64 *v73; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v74; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v75; // [rsp+48h] [rbp-B8h]
  _QWORD v76[2]; // [rsp+50h] [rbp-B0h] BYREF
  __int16 v77; // [rsp+60h] [rbp-A0h]
  __int64 v78; // [rsp+70h] [rbp-90h] BYREF
  __int64 v79; // [rsp+78h] [rbp-88h]
  char v80; // [rsp+80h] [rbp-80h]
  __int64 v81; // [rsp+84h] [rbp-7Ch]
  __int64 v82; // [rsp+8Ch] [rbp-74h]
  int v83; // [rsp+94h] [rbp-6Ch]
  int v84; // [rsp+98h] [rbp-68h]
  unsigned __int16 *v85; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v86; // [rsp+A8h] [rbp-58h]
  _BYTE v87[16]; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v88; // [rsp+C0h] [rbp-40h]

  sub_39BD220(a2, a1, a3, a4);
  v11 = *(_QWORD *)(a2 + 144);
  v79 = a2;
  v78 = a1;
  v82 = v11;
  v12 = 4 * a8;
  v83 = 4 * a8 + 8;
  v85 = (unsigned __int16 *)v87;
  v80 = 1;
  v81 = 0x148415348LL;
  v84 = 0;
  v86 = 0x400000000LL;
  if ( (unsigned __int64)(4 * a8) > 0x10 )
  {
    sub_16CD150((__int64)&v85, v87, v12 >> 2, 4, v9, v10);
    v65 = &v85[2 * (unsigned int)v86];
  }
  else
  {
    if ( !v12 )
      goto LABEL_3;
    v65 = (unsigned __int16 *)v87;
  }
  memcpy(v65, src, v12);
  LODWORD(v12) = v86;
LABEL_3:
  v88 = a5;
  LODWORD(v86) = ((4 * a8) >> 2) + v12;
  v13 = v78;
  v14 = *(_QWORD *)(v78 + 256);
  v15 = *(void (**)())(*(_QWORD *)v14 + 104LL);
  v76[0] = "Header Magic";
  v77 = 259;
  if ( v15 != nullsub_580 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v15)(v14, v76, 1);
  sub_396F340(v13, v81);
  v16 = *(_QWORD *)(v13 + 256);
  v17 = *(void (**)())(*(_QWORD *)v16 + 104LL);
  v76[0] = "Header Version";
  v77 = 259;
  if ( v17 != nullsub_580 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v17)(v16, v76, 1);
  sub_396F320(v13, WORD2(v81));
  v18 = *(_QWORD *)(v13 + 256);
  v19 = *(void (**)())(*(_QWORD *)v18 + 104LL);
  v76[0] = "Header Hash Function";
  v77 = 259;
  if ( v19 != nullsub_580 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v19)(v18, v76, 1);
  sub_396F320(v13, HIWORD(v81));
  v20 = *(_QWORD *)(v13 + 256);
  v21 = *(void (**)())(*(_QWORD *)v20 + 104LL);
  v76[0] = "Header Bucket Count";
  v77 = 259;
  if ( v21 != nullsub_580 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v21)(v20, v76, 1);
  sub_396F340(v13, v82);
  v22 = *(_QWORD *)(v13 + 256);
  v23 = *(void (**)())(*(_QWORD *)v22 + 104LL);
  v76[0] = "Header Hash Count";
  v77 = 259;
  if ( v23 != nullsub_580 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v23)(v22, v76, 1);
  sub_396F340(v13, SHIDWORD(v82));
  v24 = *(_QWORD *)(v13 + 256);
  v25 = *(void (**)())(*(_QWORD *)v24 + 104LL);
  v76[0] = "Header Data Length";
  v77 = 259;
  if ( v25 != nullsub_580 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v25)(v24, v76, 1);
  sub_396F340(v13, v83);
  v26 = v78;
  v27 = *(_QWORD *)(v78 + 256);
  v28 = *(void (**)())(*(_QWORD *)v27 + 104LL);
  v76[0] = "HeaderData Die Offset Base";
  v77 = 259;
  if ( v28 != nullsub_580 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v28)(v27, v76, 1);
  sub_396F340(v26, v84);
  v29 = *(_QWORD *)(v26 + 256);
  v30 = *(void (**)())(*(_QWORD *)v29 + 104LL);
  v76[0] = "HeaderData Atom Count";
  v77 = 259;
  if ( v30 != nullsub_580 )
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v30)(v29, v76, 1);
  sub_396F340(v26, v86);
  v31 = v85;
  v69 = &v85[2 * (unsigned int)v86];
  if ( v85 != v69 )
  {
    do
    {
      v36 = *(_QWORD *)(v26 + 256);
      v72 = *(void (**)())(*(_QWORD *)v36 + 104LL);
      v37 = sub_14E9800(*v31);
      v76[0] = &v74;
      v74 = (unsigned __int64)v37;
      v75 = v38;
      v77 = 261;
      if ( v72 != nullsub_580 )
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v72)(v36, v76, 1);
      sub_396F320(v26, *v31);
      v32 = *(_QWORD *)(v26 + 256);
      v71 = *(void (**)())(*(_QWORD *)v32 + 104LL);
      v33 = sub_14E3630(v31[1]);
      v76[0] = &v74;
      v74 = (unsigned __int64)v33;
      v75 = v34;
      v77 = 261;
      if ( v71 != nullsub_580 )
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v71)(v32, v76, 1);
      v35 = v31[1];
      v31 += 2;
      sub_396F320(v26, v35);
    }
    while ( v69 != v31 );
  }
  v39 = *(_QWORD *)(v79 + 176);
  v40 = *(_QWORD *)(v79 + 184);
  v74 = 0;
  v41 = 0xAAAAAAAAAAAAAAABLL * ((v40 - v39) >> 3);
  if ( v41 )
  {
    v42 = 0;
    do
    {
      v43 = v78;
      v44 = *(_QWORD *)(v78 + 256);
      v45 = *(void (**)())(*(_QWORD *)v44 + 104LL);
      v76[0] = "Bucket ";
      v76[1] = &v74;
      v77 = 2819;
      if ( v45 != nullsub_580 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v45)(v44, v76, 1);
        v43 = v78;
      }
      if ( *(_QWORD *)(v39 + 24 * v74 + 8) == *(_QWORD *)(v39 + 24 * v74) )
        sub_396F340(v43, -1);
      else
        sub_396F340(v43, v42);
      v46 = (__int64 *)(v39 + 24 * v74);
      v47 = *v46;
      v48 = v46[1];
      if ( *v46 != v48 )
      {
        v49 = -1;
        do
        {
          v50 = v49;
          v49 = *(unsigned int *)(*(_QWORD *)v47 + 8LL);
          v47 += 8;
          v42 += v50 != v49;
        }
        while ( v48 != v47 );
      }
      ++v74;
    }
    while ( v41 > v74 );
  }
  sub_39BAB10((__int64)&v78);
  sub_39BAC30((__int64)&v78);
  v66 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(v79 + 184) - *(_QWORD *)(v79 + 176)) >> 3);
  if ( v66 )
  {
    v67 = *(__int64 *****)(v79 + 176);
    for ( i = 0; i != v66; ++i )
    {
      v51 = *v67;
      v52 = v67[1];
      if ( *v67 != v52 )
      {
        while ( 1 )
        {
          (*(void (__fastcall **)(_QWORD, __int64 *, _QWORD))(**(_QWORD **)(v78 + 256) + 176LL))(
            *(_QWORD *)(v78 + 256),
            (*v51)[5],
            0);
          v53 = v78;
          v54 = *(_QWORD *)(v78 + 256);
          v55 = *(void (**)())(*(_QWORD *)v54 + 104LL);
          v56 = **v51;
          v57 = *v56;
          v76[0] = &v74;
          v74 = (unsigned __int64)(v56 + 3);
          v75 = v57;
          v77 = 261;
          if ( v55 != nullsub_580 )
          {
            ((void (__fastcall *)(__int64, _QWORD *, __int64))v55)(v54, v76, 1);
            v53 = v78;
          }
          sub_397C4E0(v53, (**v51)[1], (**v51)[2]);
          v58 = v78;
          v59 = *(_QWORD *)(v78 + 256);
          v60 = *(void (**)())(*(_QWORD *)v59 + 104LL);
          v76[0] = "Num DIEs";
          v77 = 259;
          if ( v60 != nullsub_580 )
          {
            ((void (__fastcall *)(__int64, _QWORD *, __int64))v60)(v59, v76, 1);
            v58 = v78;
          }
          sub_396F340(v58, (*v51)[3] - (*v51)[2]);
          v61 = *v51;
          v62 = (*v51)[2];
          v63 = (*v51)[3];
          if ( v62 != v63 )
          {
            do
            {
              v70 = v63;
              v73 = v62;
              (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)*v62 + 24LL))(*v62, v78);
              v63 = v70;
              v62 = v73 + 1;
            }
            while ( v70 != v73 + 1 );
            v61 = *v51;
          }
          ++v51;
          v64 = *((_DWORD *)v61 + 2);
          if ( v52 == v51 )
            break;
          if ( *((_DWORD *)*v51 + 2) != v64 )
            sub_396F340(v78, 0);
        }
        if ( v67[1] != *v67 )
          sub_396F340(v78, 0);
      }
      v67 += 3;
    }
  }
  if ( v85 != (unsigned __int16 *)v87 )
    _libc_free((unsigned __int64)v85);
}
