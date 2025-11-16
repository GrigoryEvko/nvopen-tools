// Function: sub_2E7B480
// Address: 0x2e7b480
//
__int64 __fastcall sub_2E7B480(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v2; // r13
  _QWORD *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdi
  unsigned __int64 v9; // rdx
  __int64 (*v10)(); // rax
  __int16 v11; // ax
  __int64 v12; // rax
  int v13; // r14d
  int v14; // r15d
  __int64 (__fastcall *v15)(__int64); // rax
  __int64 v16; // r15
  __int64 v17; // rax
  __int16 v18; // ax
  __int64 v19; // rax
  __int64 (__fastcall *v20)(__int64); // rcx
  __int64 (__fastcall *v21)(__int64); // rax
  __int64 v22; // rax
  __int64 v23; // rax
  _BYTE *v24; // r15
  _BYTE *v25; // rbx
  _BYTE *v26; // r13
  _BYTE *v27; // r15
  __int64 (__fastcall *v28)(__int64); // rax
  __int64 v29; // rax
  __int16 v30; // ax
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r14
  unsigned int v35; // eax
  __int64 v36; // r9
  _BYTE *v37; // rbx
  unsigned int v38; // r15d
  __int64 v39; // rdx
  _BYTE *v40; // r13
  __int64 v41; // r8
  __int64 v42; // rdx
  __int64 v43; // r12
  __int64 v45; // rax
  __int64 v46; // r15
  _BYTE *v47; // rdx
  _BYTE *v48; // r15
  _BYTE *v49; // rbx
  _BYTE *v50; // r14
  unsigned int v51; // edx
  _BYTE *v52; // rbx
  __int64 v53; // r14
  __int64 v54; // r14
  __int64 v55; // rax
  _QWORD *v56; // r13
  __int64 *v57; // rbx
  __int64 v58; // r14
  __int64 v59; // rdx
  __int64 v60; // rax
  int v61; // eax
  __int64 v62; // rax
  unsigned int v63; // r15d
  __int64 v64; // rbx
  __int64 v65; // r8
  __int64 v66; // r9
  _BYTE *v67; // r14
  int v68; // ecx
  __int64 v69; // rdx
  unsigned int v70; // r15d
  unsigned int v71; // eax
  __int64 v72; // r9
  _BYTE *v73; // rbx
  __int64 v74; // r8
  __int64 v75; // rdx
  __int64 v76; // rax
  __int64 (__fastcall *v77)(__int64); // rax
  __int64 v78; // [rsp+8h] [rbp-E8h]
  __int64 v79; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v80; // [rsp+20h] [rbp-D0h]
  unsigned int v81; // [rsp+28h] [rbp-C8h]
  __int64 v82; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v83; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v84; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v85; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v86; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v87; // [rsp+38h] [rbp-B8h]
  __int64 v88; // [rsp+40h] [rbp-B0h] BYREF
  unsigned __int8 *v89; // [rsp+48h] [rbp-A8h] BYREF
  _BYTE v90[32]; // [rsp+50h] [rbp-A0h] BYREF
  _BYTE *v91; // [rsp+70h] [rbp-80h] BYREF
  __int64 v92; // [rsp+78h] [rbp-78h]
  _BYTE v93[16]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v94; // [rsp+90h] [rbp-60h] BYREF
  _DWORD *v95; // [rsp+98h] [rbp-58h]
  __int64 v96; // [rsp+A0h] [rbp-50h]
  __int64 v97; // [rsp+A8h] [rbp-48h]
  __int64 v98; // [rsp+B0h] [rbp-40h]

  v2 = 0;
  v4 = *(_QWORD **)(a1 + 32);
  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v4 + 16LL) + 200LL))(*(_QWORD *)(*v4 + 16LL));
  v8 = *(_QWORD *)(a1 + 16);
  v9 = a2;
  v79 = v5;
  v10 = *(__int64 (**)())(*(_QWORD *)v8 + 128LL);
  if ( v10 != sub_2DAC790 )
  {
    v76 = ((__int64 (__fastcall *)(__int64, unsigned __int64, unsigned __int64))v10)(v8, a2, a2);
    v9 = a2;
    v2 = (_QWORD *)v76;
  }
  v11 = *(_WORD *)(v9 + 68);
  if ( v11 == 20 )
  {
    v45 = *(_QWORD *)(v9 + 32);
    v13 = *(_DWORD *)(v45 + 48);
    v14 = (*(_DWORD *)(v45 + 40) >> 8) & 0xFFF;
  }
  else if ( v11 == 12 )
  {
    v12 = *(_QWORD *)(v9 + 32);
    v13 = *(_DWORD *)(v12 + 88);
    v14 = *(_DWORD *)(v12 + 144);
  }
  else
  {
    v28 = *(__int64 (__fastcall **)(__int64))(*v2 + 520LL);
    if ( v28 != sub_2DCA430 )
    {
      v87 = v9;
      ((void (__fastcall *)(__int64 *, _QWORD *))v28)(&v94, v2);
      v9 = v87;
    }
    v13 = v95[2];
    v14 = (*v95 >> 8) & 0xFFF;
  }
  v91 = v93;
  v92 = 0x400000000LL;
  if ( v13 >= 0 )
  {
LABEL_36:
    v83 = v9;
    v30 = *(_WORD *)(v9 + 68);
    if ( v30 == 20 )
    {
      v81 = *(_DWORD *)(*(_QWORD *)(v9 + 32) + 48LL);
    }
    else if ( v30 == 12 )
    {
      v81 = *(_DWORD *)(*(_QWORD *)(v9 + 32) + 88LL);
    }
    else
    {
      v77 = *(__int64 (__fastcall **)(__int64))(*v2 + 520LL);
      if ( v77 != sub_2DCA430 )
        ((void (__fastcall *)(__int64 *, _QWORD *, unsigned __int64))v77)(&v94, v2, v9);
      v81 = v95[2];
    }
    v46 = *(_QWORD *)(v83 + 24);
    v78 = v46 + 48;
    if ( v46 + 48 != v83 )
    {
      v80 = v83;
      do
      {
        v47 = *(_BYTE **)(v80 + 32);
        v48 = &v47[40 * (*(_DWORD *)(v80 + 40) & 0xFFFFFF)];
        if ( v47 != v48 )
        {
          v49 = *(_BYTE **)(v80 + 32);
          while ( 1 )
          {
            v50 = v49;
            if ( sub_2DADC00(v49) )
              break;
            v49 += 40;
            if ( v48 == v49 )
              goto LABEL_74;
          }
          while ( v48 != v50 )
          {
            v51 = *((_DWORD *)v50 + 2);
            if ( v51 == v81
              || v81 - 1 <= 0x3FFFFFFE && v51 - 1 <= 0x3FFFFFFE && (unsigned __int8)sub_E92070(v79, v81, v51) )
            {
              v70 = sub_2E8E690(v80);
              v71 = sub_2EAB0A0(v50);
              v73 = v91;
              v34 = v70;
              v39 = v71;
              v40 = &v91[4 * (unsigned int)v92];
              if ( v91 == v40 )
                goto LABEL_55;
              while ( 1 )
              {
                v74 = *((unsigned int *)v40 - 1);
                v40 -= 4;
                v75 = (v39 << 32) | v70;
                v34 = (unsigned int)(*(_DWORD *)(a1 + 896) + 1);
                *(_DWORD *)(a1 + 896) = v34;
                v70 = v34;
                sub_2E79810(a1, v34, v75, v74, v74, v72);
                if ( v73 == v40 )
                  break;
                v39 = 0;
              }
              goto LABEL_54;
            }
            if ( v50 + 40 == v48 )
              break;
            v52 = v50 + 40;
            while ( 1 )
            {
              v50 = v52;
              if ( sub_2DADC00(v52) )
                break;
              v52 += 40;
              if ( v48 == v52 )
                goto LABEL_74;
            }
          }
        }
LABEL_74:
        v80 = *(_QWORD *)v80 & 0xFFFFFFFFFFFFFFF8LL;
      }
      while ( v78 != v80 );
      v46 = *(_QWORD *)(v83 + 24);
    }
    v53 = v2[1];
    v88 = 0;
    v94 = 0;
    v95 = 0;
    v54 = v53 - 680;
    v96 = 0;
    v55 = sub_2E311E0(v46);
    v56 = *(_QWORD **)(v46 + 32);
    v57 = (__int64 *)v55;
    v89 = (unsigned __int8 *)v94;
    if ( v94 )
      sub_B96E90((__int64)&v89, v94, 1);
    v58 = (__int64)sub_2E7B380(v56, v54, &v89, 0);
    if ( v89 )
      sub_B91220((__int64)&v89, (__int64)v89);
    sub_2E31040((__int64 *)(v46 + 40), v58);
    v59 = *v57;
    v60 = *(_QWORD *)v58;
    *(_QWORD *)(v58 + 8) = v57;
    v59 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v58 = v59 | v60 & 7;
    *(_QWORD *)(v59 + 8) = v58;
    *v57 = v58 | *v57 & 7;
    if ( v95 )
      sub_2E882B0(v58, v56);
    if ( v96 )
      sub_2E88680(v58, v56);
    if ( v94 )
      sub_B91220((__int64)&v94, v94);
    if ( v88 )
      sub_B91220((__int64)&v88, v88);
    v94 = 0;
    LODWORD(v95) = v81;
    v96 = 0;
    v97 = 0;
    v98 = 0;
    sub_2E8EAD0(v58, v56, &v94);
    v61 = *(_DWORD *)(a1 + 896);
    v94 = 1;
    v62 = (unsigned int)(v61 + 1);
    v96 = 0;
    *(_DWORD *)(a1 + 896) = v62;
    v63 = v62;
    v97 = v62;
    v64 = (unsigned int)v62;
    sub_2E8EAD0(v58, v56, &v94);
    v67 = v91;
    v40 = &v91[4 * (unsigned int)v92];
    if ( v91 != v40 )
    {
      do
      {
        v68 = *((_DWORD *)v40 - 1);
        v69 = v63;
        v40 -= 4;
        v64 = (unsigned int)(*(_DWORD *)(a1 + 896) + 1);
        *(_DWORD *)(a1 + 896) = v64;
        v63 = v64;
        sub_2E79810(a1, v64, v69, v68, v65, v66);
      }
      while ( v67 != v40 );
      v40 = v91;
    }
    v43 = v64;
    goto LABEL_56;
  }
  while ( 1 )
  {
    if ( v14 )
    {
      v31 = (unsigned int)v92;
      v32 = (unsigned int)v92 + 1LL;
      if ( v32 > HIDWORD(v92) )
      {
        sub_C8D5F0((__int64)&v91, v93, v32, 4u, v6, v7);
        v31 = (unsigned int)v92;
      }
      *(_DWORD *)&v91[4 * v31] = v14;
      LODWORD(v92) = v92 + 1;
    }
    v16 = 16LL * (v13 & 0x7FFFFFFF);
    v17 = *(_QWORD *)(v4[7] + v16 + 8);
    if ( v17 )
    {
      if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
      {
        v17 = *(_QWORD *)(v17 + 32);
        if ( v17 )
        {
          if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
            goto LABEL_106;
        }
      }
    }
    v9 = *(_QWORD *)(v17 + 16);
    v18 = *(_WORD *)(v9 + 68);
    if ( ((v18 - 12) & 0xFFF7) != 0 )
      break;
LABEL_8:
    if ( v18 == 20 )
    {
      v29 = *(_QWORD *)(v9 + 32);
      v13 = *(_DWORD *)(v29 + 48);
      v14 = (*(_DWORD *)(v29 + 40) >> 8) & 0xFFF;
      if ( v13 >= 0 )
        goto LABEL_36;
    }
    else
    {
      if ( v18 == 12 )
      {
        v33 = *(_QWORD *)(v9 + 32);
        v13 = *(_DWORD *)(v33 + 88);
        v14 = *(_DWORD *)(v33 + 144);
      }
      else
      {
        v15 = *(__int64 (__fastcall **)(__int64))(*v2 + 520LL);
        if ( v15 != sub_2DCA430 )
        {
          v86 = v9;
          ((void (__fastcall *)(__int64 *, _QWORD *))v15)(&v94, v2);
          v9 = v86;
        }
        v13 = v95[2];
        v14 = (*v95 >> 8) & 0xFFF;
      }
      if ( v13 >= 0 )
        goto LABEL_36;
    }
  }
  v19 = *v2;
  v20 = *(__int64 (__fastcall **)(__int64))(*v2 + 520LL);
  if ( v20 != sub_2DCA430 )
  {
    v84 = v9;
    ((void (__fastcall *)(_BYTE *, _QWORD *))v20)(v90, v2);
    v9 = v84;
    if ( v90[16] )
    {
LABEL_47:
      v18 = *(_WORD *)(v9 + 68);
      goto LABEL_8;
    }
    v19 = *v2;
  }
  v21 = *(__int64 (__fastcall **)(__int64))(v19 + 528);
  if ( v21 != sub_2E77FE0 )
  {
    v85 = v9;
    ((void (__fastcall *)(__int64 *, _QWORD *))v21)(&v94, v2);
    if ( (_BYTE)v96 )
    {
      v9 = v85;
      goto LABEL_47;
    }
  }
  v22 = *(_QWORD *)(v4[7] + v16 + 8);
  if ( v22 )
  {
    if ( (*(_BYTE *)(v22 + 3) & 0x10) == 0 )
    {
      v22 = *(_QWORD *)(v22 + 32);
      if ( v22 )
      {
        if ( (*(_BYTE *)(v22 + 3) & 0x10) == 0 )
LABEL_106:
          BUG();
      }
    }
  }
  v23 = *(_QWORD *)(v22 + 16);
  v24 = *(_BYTE **)(v23 + 32);
  v82 = v23;
  v25 = &v24[40 * (*(_DWORD *)(v23 + 40) & 0xFFFFFF)];
  if ( v24 == v25 )
LABEL_31:
    BUG();
  while ( 1 )
  {
    v26 = v24;
    if ( sub_2DADC00(v24) )
      break;
    v24 += 40;
    if ( v25 == v24 )
      goto LABEL_31;
  }
  while ( 1 )
  {
    if ( v25 == v26 )
      goto LABEL_31;
    if ( *((_DWORD *)v26 + 2) == v13 )
      break;
    v27 = v26 + 40;
    if ( v26 + 40 == v25 )
      goto LABEL_31;
    while ( 1 )
    {
      v26 = v27;
      if ( sub_2DADC00(v27) )
        break;
      v27 += 40;
      if ( v25 == v27 )
        goto LABEL_31;
    }
  }
  v34 = (unsigned int)sub_2E8E690(v82);
  v35 = sub_2EAB0A0(v26);
  v37 = v91;
  v38 = v34;
  v39 = v35;
  v40 = &v91[4 * (unsigned int)v92];
  if ( v91 != v40 )
  {
    while ( 1 )
    {
      v41 = *((unsigned int *)v40 - 1);
      v40 -= 4;
      v42 = (v39 << 32) | v38;
      v34 = (unsigned int)(*(_DWORD *)(a1 + 896) + 1);
      *(_DWORD *)(a1 + 896) = v34;
      v38 = v34;
      sub_2E79810(a1, v34, v42, v41, v41, v36);
      if ( v37 == v40 )
        break;
      v39 = 0;
    }
LABEL_54:
    v40 = v91;
    v39 = 0;
  }
LABEL_55:
  v43 = (v39 << 32) | v34;
LABEL_56:
  if ( v40 != v93 )
    _libc_free((unsigned __int64)v40);
  return v43;
}
