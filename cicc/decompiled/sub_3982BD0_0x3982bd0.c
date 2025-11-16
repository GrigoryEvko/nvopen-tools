// Function: sub_3982BD0
// Address: 0x3982bd0
//
void __fastcall sub_3982BD0(__int64 a1, _QWORD *a2, _BYTE *i, void (*a4)(), __int64 *a5, __int64 a6)
{
  unsigned __int64 v6; // rbx
  unsigned __int64 v8; // rdi
  __int64 v9; // rbx
  const char *v10; // rax
  const char *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // r9
  __int64 *v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  const char *v21; // rax
  __int64 v22; // r8
  __int64 v23; // rcx
  _BYTE *v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // r9
  unsigned int v27; // ecx
  unsigned __int64 v28; // r15
  unsigned __int64 v29; // rdx
  __int64 v30; // rsi
  __int64 v31; // rdx
  char v32; // dl
  int v33; // ecx
  char *v34; // rdi
  __int64 v35; // rsi
  __int64 *v36; // rdx
  _BYTE *v37; // rsi
  __int64 *v38; // rdi
  __int64 v39; // rdi
  unsigned __int64 v40; // rax
  __int64 v41; // [rsp-98h] [rbp-98h]
  __int64 v42; // [rsp-98h] [rbp-98h]
  __int64 v43; // [rsp-98h] [rbp-98h]
  void (*v44)(); // [rsp-90h] [rbp-90h]
  void (*v45)(); // [rsp-90h] [rbp-90h]
  void (*v46)(); // [rsp-90h] [rbp-90h]
  __int64 v47; // [rsp-80h] [rbp-80h] BYREF
  _BYTE v48[9]; // [rsp-71h] [rbp-71h] BYREF
  const char *v49; // [rsp-68h] [rbp-68h] BYREF
  _BYTE *v50; // [rsp-60h] [rbp-60h]
  _QWORD v51[2]; // [rsp-58h] [rbp-58h] BYREF
  __int16 v52; // [rsp-48h] [rbp-48h]

  if ( !*a2 )
    return;
  v6 = *(_QWORD *)*a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v6 )
    return;
  while ( 1 )
  {
    if ( *(_DWORD *)(v6 + 8) != 1 )
      goto LABEL_8;
    v8 = *(_QWORD *)(v6 + 16);
    if ( v8 <= 0x23 )
    {
      if ( v8 > 2 )
      {
        switch ( v8 )
        {
          case 3uLL:
          case 0x10uLL:
          case 0x23uLL:
            v42 = *(_QWORD *)(a1 + 256);
            v45 = *(void (**)())(*(_QWORD *)v42 + 104LL);
            v11 = sub_14E3970(v8);
            v14 = (__int64)v45;
            v51[0] = &v49;
            v15 = v42;
            v49 = v11;
            v50 = (_BYTE *)v12;
            v52 = 261;
            if ( v45 != nullsub_580 )
              ((void (__fastcall *)(__int64, _QWORD *, __int64))v45)(v42, v51, 1);
            sub_3982B10(v6 + 8, a1, v12, v14, v13, v15);
            v16 = (__int64 *)(*(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL);
            sub_3982B10((__int64)(v16 + 1), a1, v17, v18, v19, v20);
            v9 = *v16;
            if ( (v9 & 4) == 0 )
              goto LABEL_9;
            break;
          case 6uLL:
          case 0x12uLL:
          case 0x18uLL:
          case 0x22uLL:
            v41 = *(_QWORD *)(a1 + 256);
            v44 = *(void (**)())(*(_QWORD *)v41 + 104LL);
            v10 = sub_14E3970(v8);
            v51[0] = &v49;
            a6 = v41;
            v52 = 261;
            a4 = v44;
            v49 = v10;
            v50 = i;
            if ( v44 != nullsub_580 )
              ((void (__fastcall *)(__int64, _QWORD *, __int64))v44)(v41, v51, 1);
            goto LABEL_8;
          default:
            goto LABEL_8;
        }
        return;
      }
LABEL_8:
      sub_3982B10(v6 + 8, a1, (__int64)i, (__int64)a4, (__int64)a5, a6);
      v9 = *(_QWORD *)v6;
      if ( (v9 & 4) != 0 )
        return;
      goto LABEL_9;
    }
    if ( (v8 & 0xFFFFFFFFFFFFFFFDLL) != 0x90 )
      goto LABEL_8;
    v43 = *(_QWORD *)(a1 + 256);
    v46 = *(void (**)())(*(_QWORD *)v43 + 104LL);
    v21 = sub_14E3970(v8);
    v23 = (__int64)v46;
    v51[0] = &v49;
    v50 = v24;
    v25 = 261;
    v49 = v21;
    v26 = v43;
    v52 = 261;
    if ( v46 != nullsub_580 )
      ((void (__fastcall *)(__int64, _QWORD *, __int64))v46)(v43, v51, 1);
    sub_3982B10(v6 + 8, a1, v25, v23, v22, v26);
    v27 = 0;
    v28 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
    v29 = v28;
    do
    {
      if ( *(_WORD *)(v29 + 14) != 11 )
        break;
      if ( *(_DWORD *)(v29 + 8) == 1 )
      {
        v30 = v27++;
        *((_BYTE *)&v49 + v30) = *(_QWORD *)(v29 + 16);
      }
      v31 = *(_QWORD *)v29;
      if ( (v31 & 4) != 0 )
        break;
      v29 = v31 & 0xFFFFFFFFFFFFFFF8LL;
    }
    while ( v29 );
    v32 = (char)v49;
    a6 = 0;
    v33 = 0;
    v34 = (char *)&v49;
    v35 = (unsigned __int8)v49 & 0x7F;
    do
    {
      if ( v35 != (unsigned __int64)(v35 << v33) >> v33 )
        break;
      a6 += v35 << v33;
      v33 += 7;
      ++v34;
      if ( v32 >= 0 )
        goto LABEL_31;
      v32 = *v34;
      v35 = *v34 & 0x7F;
    }
    while ( v33 != 70 );
    a6 = 0;
LABEL_31:
    v47 = a6;
    a5 = &v47;
    v36 = (__int64 *)((char *)&v47 + 7);
    v37 = v48;
    do
    {
      *v37++ = *(_BYTE *)v36;
      v38 = v36;
      v36 = (__int64 *)((char *)v36 - 1);
    }
    while ( &v47 != v38 );
    v48[8] = 0;
    for ( i = v48; !*i; ++i )
      ;
    v39 = *(_QWORD *)(a1 + 256);
    a4 = *(void (**)())(*(_QWORD *)v39 + 104LL);
    v51[1] = " [unsigned LEB]";
    v51[0] = i;
    v52 = 771;
    if ( a4 != nullsub_580 )
      ((void (__fastcall *)(__int64, _QWORD *, __int64, void (*)(), __int64 *))a4)(v39, v51, 1, a4, &v47);
    if ( *(_WORD *)(v28 + 14) == 11 )
      break;
LABEL_41:
    if ( *(_QWORD *)(v6 + 16) != 146 )
    {
      v6 = v28;
      goto LABEL_10;
    }
    sub_3982B10(v28 + 8, a1, (__int64)i, (__int64)a4, (__int64)a5, a6);
    v9 = *(_QWORD *)v28;
    if ( (*(_QWORD *)v28 & 4) != 0 )
      return;
LABEL_9:
    v6 = v9 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_10:
    if ( !v6 )
      return;
  }
  while ( 1 )
  {
    sub_3982B10(v28 + 8, a1, (__int64)i, (__int64)a4, (__int64)a5, a6);
    if ( (*(_QWORD *)v28 & 4) != 0 )
      break;
    v40 = *(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL;
    v28 = v40;
    if ( !v40 )
      break;
    if ( *(_WORD *)(v40 + 14) != 11 )
      goto LABEL_41;
  }
}
