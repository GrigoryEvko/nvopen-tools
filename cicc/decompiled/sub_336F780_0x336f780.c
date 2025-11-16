// Function: sub_336F780
// Address: 0x336f780
//
char *__fastcall sub_336F780(_QWORD *a1, int a2, __int64 a3, __int64 a4)
{
  int v5; // r13d
  __int64 v7; // r14
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned __int16 ***v12; // rdx
  unsigned __int16 ***v13; // r10
  __int64 v14; // r9
  int v15; // r15d
  unsigned __int16 v16; // cx
  unsigned __int16 *v17; // rax
  unsigned __int16 v18; // di
  unsigned __int16 v19; // dx
  __int64 v20; // r8
  unsigned __int16 v21; // cx
  unsigned __int16 ***v22; // r10
  __int64 v23; // r9
  __int16 v24; // r13
  __int64 (__fastcall *v25)(__int64, __int64, _QWORD, _QWORD, char *); // rax
  unsigned int v26; // eax
  __int64 v27; // rdx
  unsigned __int16 **v28; // rax
  __int64 v29; // rdi
  unsigned __int16 *v30; // r12
  __int64 v31; // rax
  unsigned __int16 *v32; // rsi
  __int64 v33; // rcx
  __int64 v34; // rax
  unsigned __int16 *v35; // rcx
  int v36; // ebx
  __int64 v37; // r15
  unsigned __int16 *v38; // r13
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  char *v57; // rdi
  char *v59; // rax
  char *v60; // rdx
  int v61; // eax
  unsigned __int16 v62; // ax
  unsigned __int16 v63; // r11
  __int64 v64; // rax
  int v65; // edx
  __int64 v66; // rax
  int v67; // edx
  __int64 v68; // [rsp+8h] [rbp-178h]
  unsigned int v69; // [rsp+10h] [rbp-170h]
  __int64 v70; // [rsp+10h] [rbp-170h]
  unsigned __int16 ***v71; // [rsp+10h] [rbp-170h]
  __int64 v72; // [rsp+10h] [rbp-170h]
  __int64 v73; // [rsp+18h] [rbp-168h]
  __int16 v74; // [rsp+18h] [rbp-168h]
  unsigned __int16 ***v75; // [rsp+18h] [rbp-168h]
  unsigned __int16 v76; // [rsp+18h] [rbp-168h]
  unsigned __int16 ***v77; // [rsp+18h] [rbp-168h]
  unsigned __int16 v78; // [rsp+20h] [rbp-160h]
  unsigned __int16 ***v80; // [rsp+28h] [rbp-158h]
  __int64 v81; // [rsp+28h] [rbp-158h]
  __int64 v82; // [rsp+30h] [rbp-150h]
  __int64 v83; // [rsp+68h] [rbp-118h]
  _BYTE *v84; // [rsp+70h] [rbp-110h] BYREF
  __int64 v85; // [rsp+78h] [rbp-108h]
  _BYTE v86[16]; // [rsp+80h] [rbp-100h] BYREF
  char *v87[2]; // [rsp+90h] [rbp-F0h] BYREF
  char v88; // [rsp+A0h] [rbp-E0h] BYREF
  char *v89; // [rsp+E0h] [rbp-A0h] BYREF
  char v90; // [rsp+F8h] [rbp-88h] BYREF
  char *v91; // [rsp+100h] [rbp-80h] BYREF
  char v92; // [rsp+110h] [rbp-70h] BYREF
  char *v93; // [rsp+120h] [rbp-60h] BYREF
  char v94; // [rsp+130h] [rbp-50h] BYREF
  __int64 v95; // [rsp+140h] [rbp-40h]

  v5 = (int)a1;
  v7 = a1[5];
  v8 = a1[2];
  v82 = a1[8];
  v9 = *(_QWORD *)(v7 + 16);
  v84 = v86;
  v85 = 0x400000000LL;
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 200LL))(v9);
  if ( (unsigned int)(*(_DWORD *)(a3 + 224) - 2) <= 1 )
    goto LABEL_38;
  v11 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v8 + 2496LL))(
          v8,
          v10,
          *(_QWORD *)(a4 + 192),
          *(_QWORD *)(a4 + 200),
          *(unsigned __int16 *)(a4 + 240));
  v13 = v12;
  v14 = v11;
  v15 = v11;
  if ( !v12 )
    goto LABEL_38;
  v16 = *(_WORD *)(a3 + 240);
  v17 = (unsigned __int16 *)(*(_QWORD *)(v10 + 320)
                           + 2LL
                           * *(unsigned int *)(*(_QWORD *)(v10 + 312)
                                             + 16LL
                                             * (*((unsigned __int16 *)*v12 + 12)
                                              + *(_DWORD *)(v10 + 328)
                                              * (unsigned int)((__int64)(*(_QWORD *)(v10 + 288) - *(_QWORD *)(v10 + 280)) >> 3))
                                             + 12));
  v18 = *v17;
  v78 = *v17;
  if ( v16 != 1 && v18 != 264 && *(_DWORD *)a3 <= 1u )
  {
    if ( v18 == 1 )
      goto LABEL_77;
    v19 = *v17;
    do
    {
      if ( v16 == v19 )
        goto LABEL_10;
      v19 = v17[1];
      ++v17;
    }
    while ( v19 != 1 );
    if ( !v16 || (unsigned __int16)(v16 - 504) <= 7u || v78 <= 0x1FFu && (unsigned __int16)(v78 - 1) > 0x1F6u )
LABEL_77:
      BUG();
    if ( *(_QWORD *)&byte_444C4A0[16 * v78 - 16] == *(_QWORD *)&byte_444C4A0[16 * v16 - 16]
      && byte_444C4A0[16 * v78 - 8] == byte_444C4A0[16 * v16 - 8] )
    {
      if ( !*(_DWORD *)a3 && !*(_BYTE *)(a3 + 10) )
      {
        v72 = v14;
        v77 = v13;
        v66 = sub_33FAF80(v5, 234, a2, v78, 0, v14);
        v14 = v72;
        v13 = v77;
        *(_QWORD *)(a3 + 248) = v66;
        *(_DWORD *)(a3 + 256) = v67;
      }
      *(_WORD *)(a3 + 240) = v78;
    }
    else if ( ((unsigned __int16)(v78 - 2) <= 7u
            || (unsigned __int16)(v78 - 17) <= 0x6Cu
            || (unsigned __int16)(v78 - 176) <= 0x1Fu)
           && ((unsigned __int16)(v16 - 10) <= 6u
            || (unsigned __int16)(v16 - 126) <= 0x31u
            || (unsigned __int16)(v16 - 208) <= 0x14u) )
    {
      v70 = v14;
      v75 = v13;
      v59 = (char *)sub_3368600((_WORD *)(a3 + 240));
      v87[1] = v60;
      v87[0] = v59;
      v61 = sub_CA1930(v87);
      v62 = sub_3368630(v61);
      v13 = v75;
      v14 = v70;
      v63 = v62;
      if ( !*(_DWORD *)a3 )
      {
        v68 = v70;
        v71 = v75;
        v76 = v62;
        v64 = sub_33FAF80(v5, 234, a2, v62, 0, v14);
        v14 = v68;
        v13 = v71;
        v63 = v76;
        *(_QWORD *)(a3 + 248) = v64;
        *(_DWORD *)(a3 + 256) = v65;
      }
      *(_WORD *)(a3 + 240) = v63;
    }
  }
LABEL_10:
  v73 = v14;
  v80 = v13;
  if ( (unsigned __int8)sub_344B5F0(a3) )
    goto LABEL_38;
  v21 = *(_WORD *)(a3 + 240);
  v22 = v80;
  v23 = v73;
  v24 = v21;
  if ( v21 == 1 )
  {
    v28 = *v80;
    v27 = 1;
    v24 = v78;
    v30 = **v80;
    v29 = *(_QWORD *)(v7 + 32);
    if ( !(_DWORD)v73 )
      goto LABEL_22;
  }
  else
  {
    v25 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD, _QWORD, char *))(*(_QWORD *)v8 + 736LL);
    LOWORD(v87[0]) = v78;
    BYTE2(v87[0]) = 1;
    v26 = v25(v8, v82, v21, 0, v87[0]);
    v22 = v80;
    v27 = v26;
    v23 = v73;
    v28 = *v80;
    v29 = *(_QWORD *)(v7 + 32);
    v30 = **v80;
    if ( !(_DWORD)v73 )
      goto LABEL_21;
  }
  v31 = 2LL * *((unsigned __int16 *)v28 + 10);
  v32 = &v30[(unsigned __int64)v31 / 2];
  v33 = v31 >> 1;
  v34 = v31 >> 3;
  if ( !v34 )
  {
LABEL_46:
    if ( v33 != 2 )
    {
      if ( v33 != 3 )
      {
        if ( v33 != 1 )
          goto LABEL_50;
        goto LABEL_49;
      }
      if ( (_DWORD)v23 == *v30 )
        goto LABEL_20;
      ++v30;
    }
    if ( (_DWORD)v23 == *v30 )
      goto LABEL_20;
    ++v30;
LABEL_49:
    if ( (_DWORD)v23 != *v30 )
      goto LABEL_50;
    goto LABEL_20;
  }
  v35 = &v30[4 * v34];
  while ( v15 != *v30 )
  {
    if ( v15 == v30[1] )
    {
      ++v30;
      break;
    }
    if ( v15 == v30[2] )
    {
      v30 += 2;
      break;
    }
    if ( v15 == v30[3] )
    {
      v30 += 3;
      break;
    }
    v30 += 4;
    if ( v35 == v30 )
    {
      v33 = v32 - v30;
      goto LABEL_46;
    }
  }
LABEL_20:
  if ( v32 != v30 )
  {
LABEL_21:
    if ( !(_DWORD)v27 )
    {
LABEL_30:
      BYTE4(v83) = 0;
      sub_336F670((__int64)v87, (__int64)&v84, v78, v24, 0, v83);
      sub_3365840(a3 + 264, v87, v41, v42, v43, v44);
      sub_33656C0(a3 + 344, &v89, v45, v46, v47, v48);
      sub_3365560(a3 + 376, &v91, v49, v50, v51, v52);
      sub_33659A0(a3 + 408, &v93, v53, v54, v55, v56);
      v57 = v93;
      *(_QWORD *)(a3 + 440) = v95;
      if ( v57 != &v94 )
        _libc_free((unsigned __int64)v57);
      if ( v91 != &v92 )
        _libc_free((unsigned __int64)v91);
      if ( v89 != &v90 )
        _libc_free((unsigned __int64)v89);
      if ( v87[0] != &v88 )
        _libc_free((unsigned __int64)v87[0]);
LABEL_38:
      BYTE4(v87[0]) = 0;
      goto LABEL_39;
    }
LABEL_22:
    v81 = a3;
    v36 = v15;
    v74 = v24;
    v37 = (__int64)v22;
    v38 = &v30[v27];
    do
    {
      if ( v36 )
        v20 = *v30;
      else
        v20 = (unsigned int)sub_2EC06C0(v29, v37, byte_3F871B3, 0, v20, v23);
      v39 = (unsigned int)v85;
      v40 = (unsigned int)v85 + 1LL;
      if ( v40 > HIDWORD(v85) )
      {
        v69 = v20;
        sub_C8D5F0((__int64)&v84, v86, v40, 4u, v20, v23);
        v39 = (unsigned int)v85;
        v20 = v69;
      }
      ++v30;
      *(_DWORD *)&v84[4 * v39] = v20;
      LODWORD(v85) = v85 + 1;
    }
    while ( v30 != v38 );
    a3 = v81;
    v24 = v74;
    goto LABEL_30;
  }
LABEL_50:
  LODWORD(v87[0]) = v23;
  BYTE4(v87[0]) = 1;
LABEL_39:
  if ( v84 != v86 )
    _libc_free((unsigned __int64)v84);
  return v87[0];
}
