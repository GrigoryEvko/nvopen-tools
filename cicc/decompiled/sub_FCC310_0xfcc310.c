// Function: sub_FCC310
// Address: 0xfcc310
//
void __fastcall sub_FCC310(_DWORD *a1, __int64 a2)
{
  void *v3; // rsi
  __int64 v4; // r14
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r14
  __int64 v15; // rsi
  unsigned __int8 *v16; // rsi
  __int64 v17; // r14
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rsi
  __int64 v28; // rsi
  unsigned __int8 *v29; // rsi
  __int64 v30; // r14
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rsi
  __int64 v41; // rsi
  unsigned __int8 *v42; // rsi
  __int64 v43; // rsi
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  unsigned __int64 v47; // r9
  __int64 v48; // rdx
  void *v49; // r8
  char *v50; // r14
  __int64 v51; // rcx
  __int64 *v52; // r15
  __int64 v53; // rax
  unsigned int v54; // eax
  _BYTE *v55; // r13
  __int64 v56; // rbx
  _BYTE *v57; // r14
  int v58; // eax
  __int64 v59; // rdx
  _QWORD *v60; // rax
  unsigned int v61; // ebx
  __int64 v62; // rax
  char *v63; // rdx
  unsigned __int8 *v64; // rsi
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rax
  _QWORD *v70; // r14
  __int64 v71; // r14
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 *v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rsi
  unsigned int v82; // [rsp+0h] [rbp-F0h]
  __int64 v83; // [rsp+0h] [rbp-F0h]
  int v84; // [rsp+Ch] [rbp-E4h]
  void *v85; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v86; // [rsp+28h] [rbp-C8h] BYREF
  __int64 *v87; // [rsp+30h] [rbp-C0h]
  __int64 v88; // [rsp+38h] [rbp-B8h]
  __int64 *v89; // [rsp+40h] [rbp-B0h]
  __int64 v90; // [rsp+48h] [rbp-A8h]
  __int64 *v91; // [rsp+50h] [rbp-A0h]
  __int64 v92; // [rsp+58h] [rbp-98h]
  void *s1; // [rsp+60h] [rbp-90h] BYREF
  __int64 v94; // [rsp+68h] [rbp-88h]
  _BYTE v95[32]; // [rsp+70h] [rbp-80h] BYREF
  void *s2; // [rsp+90h] [rbp-60h] BYREF
  __int64 v97; // [rsp+98h] [rbp-58h]
  _BYTE v98[80]; // [rsp+A0h] [rbp-50h] BYREF

  v3 = *(void **)(a2 + 24);
  s2 = v3;
  if ( v3 )
    sub_B96E90((__int64)&s2, (__int64)v3, 1);
  v4 = sub_B10CD0((__int64)&s2);
  v9 = sub_FC95E0((__int64)a1, v4, v5, v6, v7, v8);
  v88 = v10;
  v87 = v9;
  if ( (_BYTE)v10 )
    v14 = (__int64)v87;
  else
    v14 = sub_FCBA10((__int64)a1, v4, v10, v11, v12, v13);
  if ( s2 )
    sub_B91220((__int64)&s2, (__int64)s2);
  sub_B10CB0(&s2, v14);
  v15 = *(_QWORD *)(a2 + 24);
  if ( v15 )
    sub_B91220(a2 + 24, v15);
  v16 = (unsigned __int8 *)s2;
  *(_QWORD *)(a2 + 24) = s2;
  if ( v16 )
    sub_B976B0((__int64)&s2, v16, a2 + 24);
  if ( *(_BYTE *)(a2 + 32) == 1 )
  {
    v17 = sub_B11FB0(a2 + 40);
    v22 = sub_FC95E0((__int64)a1, v17, v18, v19, v20, v21);
    v97 = v23;
    s2 = v22;
    if ( (_BYTE)v23 )
      v27 = (__int64)s2;
    else
      v27 = sub_FCBA10((__int64)a1, v17, v23, v24, v25, v26);
    sub_B11F70(&s1, v27);
    v28 = *(_QWORD *)(a2 + 40);
    if ( v28 )
      sub_B91220(a2 + 40, v28);
    v29 = (unsigned __int8 *)s1;
    *(_QWORD *)(a2 + 40) = s1;
    if ( v29 )
      sub_B976B0((__int64)&s1, v29, a2 + 40);
    return;
  }
  v30 = sub_B12000(a2 + 72);
  v35 = sub_FC95E0((__int64)a1, v30, v31, v32, v33, v34);
  v90 = v36;
  v89 = v35;
  if ( (_BYTE)v36 )
    v40 = (__int64)v89;
  else
    v40 = sub_FCBA10((__int64)a1, v30, v36, v37, v38, v39);
  sub_B11FC0(&s2, v40);
  v41 = *(_QWORD *)(a2 + 72);
  if ( v41 )
    sub_B91220(a2 + 72, v41);
  v42 = (unsigned __int8 *)s2;
  *(_QWORD *)(a2 + 72) = s2;
  if ( v42 )
    sub_B976B0((__int64)&s2, v42, a2 + 72);
  v84 = *a1 & 2;
  if ( *(_BYTE *)(a2 + 64) == 2 )
  {
    v64 = sub_B13320(a2);
    v69 = sub_FC8800((__int64)a1, (__int64)v64, v65, v66, v67, v68);
    if ( v84 )
    {
      if ( !v69 )
        goto LABEL_62;
    }
    else if ( !v69 )
    {
      sub_B14010(a2, (__int64)v64);
      goto LABEL_62;
    }
    v70 = sub_B98A20(v69, (__int64)v64);
    sub_B91340(a2 + 40, 1);
    *(_QWORD *)(a2 + 48) = v70;
    sub_B96F50(a2 + 40, 1);
LABEL_62:
    v71 = sub_B13870(a2);
    v76 = sub_FC95E0((__int64)a1, v71, v72, v73, v74, v75);
    v92 = v77;
    v91 = v76;
    if ( (_BYTE)v77 )
      v81 = (__int64)v91;
    else
      v81 = sub_FCBA10((__int64)a1, v71, v77, v78, v79, v80);
    sub_B13D10(a2, v81);
  }
  sub_B129C0(&s2, a2);
  v94 = 0x400000000LL;
  v43 = (__int64)&v85;
  s1 = v95;
  v86 = v97;
  v85 = s2;
  sub_FC7C70((__int64)&s1, (__int64 *)&v85, &v86, v44, v45, v46);
  v48 = (unsigned int)v94;
  v49 = s1;
  v97 = 0x400000000LL;
  v50 = (char *)s1 + 8 * (unsigned int)v94;
  v51 = (unsigned int)v94;
  s2 = v98;
  if ( v50 == s1 )
  {
    v55 = v98;
    v56 = 0;
  }
  else
  {
    v52 = (__int64 *)s1;
    do
    {
      v43 = *v52;
      v53 = sub_FC8800((__int64)a1, *v52, v48, v51, (__int64)v49, v47);
      v48 = (unsigned int)v97;
      v47 = (unsigned int)v97 + 1LL;
      if ( v47 > HIDWORD(v97) )
      {
        v43 = (__int64)v98;
        v83 = v53;
        sub_C8D5F0((__int64)&s2, v98, (unsigned int)v97 + 1LL, 8u, (__int64)v49, v47);
        v48 = (unsigned int)v97;
        v53 = v83;
      }
      v51 = (__int64)s2;
      ++v52;
      *((_QWORD *)s2 + v48) = v53;
      v54 = v97 + 1;
      LODWORD(v97) = v97 + 1;
    }
    while ( v50 != (char *)v52 );
    v48 = (unsigned int)v94;
    v55 = s2;
    v56 = v54;
    v51 = (unsigned int)v94;
  }
  if ( v48 != v56
    || (v82 = v51, v57 = s1, 8 * v56) && (v43 = (__int64)v55, v58 = memcmp(s1, v55, 8 * v56), v51 = v82, v58) )
  {
    if ( v84 )
      goto LABEL_49;
    v43 = (__int64)&v55[8 * v56];
    v59 = (8 * v56) >> 3;
    if ( (8 * v56) >> 5 )
    {
      v60 = v55;
      while ( *v60 )
      {
        if ( !v60[1] )
        {
          ++v60;
          break;
        }
        if ( !v60[2] )
        {
          v60 += 2;
          break;
        }
        if ( !v60[3] )
        {
          v60 += 3;
          break;
        }
        v60 += 4;
        if ( &v55[32 * ((8 * v56) >> 5)] == (_BYTE *)v60 )
        {
          v59 = (v43 - (__int64)v60) >> 3;
          goto LABEL_75;
        }
      }
LABEL_48:
      if ( (_QWORD *)v43 != v60 )
      {
        sub_B13710(a2);
        v55 = s2;
LABEL_53:
        if ( v55 != v98 )
          _libc_free(v55, v43);
        if ( s1 != v95 )
          _libc_free(s1, v43);
        return;
      }
LABEL_49:
      v61 = 0;
      v62 = 0;
      if ( (_DWORD)v51 )
      {
        do
        {
          v63 = *(char **)&v55[8 * v62];
          if ( v63 )
          {
            v43 = v61;
            sub_B12AA0(a2, v61, v63, v51);
            v55 = s2;
          }
          v62 = v61 + 1;
          v61 = v62;
        }
        while ( (unsigned int)v62 < (unsigned int)v94 );
      }
      goto LABEL_53;
    }
    v60 = v55;
LABEL_75:
    if ( v59 != 2 )
    {
      if ( v59 != 3 )
      {
        if ( v59 != 1 )
          goto LABEL_49;
        goto LABEL_78;
      }
      if ( !*v60 )
        goto LABEL_48;
      ++v60;
    }
    if ( !*v60 )
      goto LABEL_48;
    ++v60;
LABEL_78:
    if ( *v60 )
      goto LABEL_49;
    goto LABEL_48;
  }
  if ( v55 != v98 )
  {
    _libc_free(v55, v43);
    v57 = s1;
  }
  if ( v57 != v95 )
    _libc_free(v57, v43);
}
