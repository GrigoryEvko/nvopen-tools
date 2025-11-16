// Function: sub_F377A0
// Address: 0xf377a0
//
__int64 __fastcall sub_F377A0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 *a4,
        __int64 *a5,
        char a6,
        char a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  _QWORD *v10; // r15
  __int64 *v11; // r13
  unsigned __int64 v12; // rax
  __int64 v13; // r12
  int v14; // r14d
  char v15; // bl
  unsigned int v16; // r13d
  __int64 *v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // r12
  char v25; // bl
  __int64 v26; // r15
  unsigned __int64 v27; // r13
  unsigned __int8 *v28; // rax
  unsigned __int8 *v29; // r12
  unsigned __int8 *v30; // rsi
  __int64 result; // rax
  __int64 v32; // r8
  __int64 *v33; // r8
  __int64 *v34; // r9
  __int64 *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 *v38; // r15
  __int64 *v39; // rsi
  __int64 v40; // rdx
  __int64 *v41; // r14
  __int64 *v42; // r15
  __int64 v43; // rdx
  __int64 *v44; // r14
  int v45; // ecx
  __int64 v46; // rdi
  int v47; // ecx
  unsigned int v48; // edx
  _QWORD *v49; // r8
  __int64 *v50; // r13
  unsigned __int8 *v51; // rax
  __int64 *v52; // rax
  unsigned __int8 **v53; // rax
  __int64 *v54; // rax
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // r15
  _QWORD *v58; // rdi
  unsigned __int64 v59; // rax
  __int64 v60; // rdx
  unsigned __int64 v61; // r15
  __int64 *v62; // r8
  __int64 v63; // rax
  __int64 v64; // r15
  _QWORD *v65; // rdi
  unsigned __int64 v66; // rax
  __int64 v67; // rsi
  unsigned __int64 v68; // rcx
  __int64 *v69; // r13
  __int64 v70; // rsi
  unsigned __int8 *v71; // rsi
  __int64 v72; // rsi
  unsigned __int8 *v73; // rsi
  unsigned int v74; // r9d
  __int64 v78; // [rsp+10h] [rbp-1A0h]
  __int64 *v79; // [rsp+10h] [rbp-1A0h]
  unsigned __int64 v80; // [rsp+10h] [rbp-1A0h]
  unsigned __int64 v81; // [rsp+10h] [rbp-1A0h]
  __int64 *v82; // [rsp+10h] [rbp-1A0h]
  char v85; // [rsp+28h] [rbp-188h]
  __int64 v86; // [rsp+28h] [rbp-188h]
  _QWORD *v88; // [rsp+40h] [rbp-170h] BYREF
  unsigned __int8 *v89; // [rsp+48h] [rbp-168h] BYREF
  _QWORD *v90; // [rsp+50h] [rbp-160h] BYREF
  _QWORD *v91; // [rsp+58h] [rbp-158h] BYREF
  unsigned __int8 *v92; // [rsp+60h] [rbp-150h] BYREF
  __int64 v93; // [rsp+68h] [rbp-148h]
  __int16 v94; // [rsp+80h] [rbp-130h]
  __int64 v95; // [rsp+90h] [rbp-120h] BYREF
  __int64 *v96; // [rsp+98h] [rbp-118h]
  __int64 v97; // [rsp+A0h] [rbp-110h]
  int v98; // [rsp+A8h] [rbp-108h]
  char v99; // [rsp+ACh] [rbp-104h]
  char v100; // [rsp+B0h] [rbp-100h] BYREF
  unsigned __int8 *v101; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v102; // [rsp+F8h] [rbp-B8h]
  _BYTE v103[176]; // [rsp+100h] [rbp-B0h] BYREF

  v101 = v103;
  v102 = 0x800000000LL;
  v95 = 0;
  v96 = (__int64 *)&v100;
  v97 = 8;
  v98 = 0;
  v99 = 1;
  if ( !a2 )
    BUG();
  v10 = (_QWORD *)a2[2];
  v11 = a2;
  v88 = v10;
  if ( a9 )
  {
    v12 = sub_986580((__int64)v10);
    v13 = v12;
    if ( v12 )
    {
      v14 = sub_B46E30(v12);
      if ( v14 )
      {
        v15 = 1;
        v16 = 0;
        while ( 1 )
        {
          while ( 1 )
          {
            v19 = sub_B46EC0(v13, v16);
            if ( v15 )
              break;
LABEL_60:
            ++v16;
            sub_C8CC70((__int64)&v95, v19, (__int64)v17, v18, v20, v21);
            v15 = v99;
            if ( v14 == v16 )
              goto LABEL_12;
          }
          v22 = v96;
          v17 = &v96[HIDWORD(v97)];
          if ( v96 == v17 )
          {
LABEL_62:
            if ( HIDWORD(v97) >= (unsigned int)v97 )
              goto LABEL_60;
            ++v16;
            ++HIDWORD(v97);
            *v17 = v19;
            v15 = v99;
            ++v95;
            if ( v14 == v16 )
              goto LABEL_12;
          }
          else
          {
            while ( v19 != *v22 )
            {
              if ( v17 == ++v22 )
                goto LABEL_62;
            }
            if ( v14 == ++v16 )
            {
LABEL_12:
              v11 = a2;
              v23 = (unsigned int)(2 * (HIDWORD(v97) - v98) + 4);
              if ( HIDWORD(v102) < (unsigned int)v23 )
                sub_C8D5F0((__int64)&v101, v103, v23, 0x10u, v20, v21);
              v10 = v88;
              break;
            }
          }
        }
      }
    }
  }
  v94 = 257;
  v24 = sub_AA48A0((__int64)v10);
  v89 = (unsigned __int8 *)sub_AA8550(v88, v11, a3, (__int64)&v92, 0);
  v90 = v89;
  v91 = v89;
  if ( a4 )
  {
    if ( *a4 )
    {
      v90 = (_QWORD *)*a4;
      v25 = 0;
      goto LABEL_18;
    }
    v55 = v88[9];
    v94 = 257;
    v78 = v55;
    v56 = sub_22077B0(80);
    v57 = v56;
    if ( v56 )
      sub_AA4D50(v56, v24, (__int64)&v92, v78, (__int64)v89);
    v90 = (_QWORD *)v57;
    if ( a6 )
    {
      sub_B43C20((__int64)&v92, v57);
      v58 = sub_BD2C40(72, unk_3F148B8);
      if ( v58 )
        sub_B4C8A0((__int64)v58, v24, (__int64)v92, v93);
      v25 = 0;
    }
    else
    {
      sub_B43C20((__int64)&v92, v57);
      v25 = 1;
      sub_F340F0((__int64)v89, (__int64)v92, v93);
    }
    v59 = sub_986580((__int64)v90);
    v60 = (__int64)v90;
    v61 = v59;
    v62 = (__int64 *)(v59 + 48);
    v92 = (unsigned __int8 *)v11[3];
    if ( v92 )
    {
      v79 = (__int64 *)(v59 + 48);
      sub_B96E90((__int64)&v92, (__int64)v92, 1);
      v62 = v79;
      if ( v79 == (__int64 *)&v92 )
      {
        if ( v92 )
          sub_B91220((__int64)&v92, (__int64)v92);
        goto LABEL_95;
      }
      v72 = *(_QWORD *)(v61 + 48);
      if ( !v72 )
      {
LABEL_119:
        v73 = v92;
        *(_QWORD *)(v61 + 48) = v92;
        if ( v73 )
        {
          sub_B976B0((__int64)&v92, v73, (__int64)v62);
          v60 = (__int64)v90;
          goto LABEL_96;
        }
LABEL_95:
        v60 = (__int64)v90;
LABEL_96:
        *a4 = v60;
LABEL_18:
        if ( a5 )
          goto LABEL_19;
LABEL_84:
        v85 = 0;
        v26 = (__int64)v91;
        goto LABEL_21;
      }
    }
    else
    {
      if ( v62 == (__int64 *)&v92 )
        goto LABEL_96;
      v72 = *(_QWORD *)(v59 + 48);
      if ( !v72 )
        goto LABEL_96;
    }
    v82 = v62;
    sub_B91220((__int64)v62, v72);
    v62 = v82;
    goto LABEL_119;
  }
  v25 = 0;
  if ( !a5 )
    goto LABEL_84;
LABEL_19:
  v26 = *a5;
  if ( *a5 )
  {
    v91 = (_QWORD *)*a5;
    v85 = 0;
    goto LABEL_21;
  }
  v86 = v88[9];
  v94 = 257;
  v63 = sub_22077B0(80);
  v64 = v63;
  if ( v63 )
    sub_AA4D50(v63, v24, (__int64)&v92, v86, (__int64)v89);
  v91 = (_QWORD *)v64;
  if ( a7 )
  {
    sub_B43C20((__int64)&v92, v64);
    v65 = sub_BD2C40(72, unk_3F148B8);
    if ( v65 )
      sub_B4C8A0((__int64)v65, v24, (__int64)v92, v93);
    v85 = 0;
  }
  else
  {
    sub_B43C20((__int64)&v92, v64);
    sub_F340F0((__int64)v89, (__int64)v92, v93);
    v85 = 1;
  }
  v26 = (__int64)v91;
  v66 = sub_986580((__int64)v91);
  v67 = v11[3];
  v68 = v66;
  v69 = (__int64 *)(v66 + 48);
  v92 = (unsigned __int8 *)v67;
  if ( !v67 )
  {
    if ( v69 == (__int64 *)&v92 )
      goto LABEL_108;
    v70 = *(_QWORD *)(v66 + 48);
    if ( !v70 )
      goto LABEL_108;
LABEL_113:
    v81 = v68;
    sub_B91220((__int64)v69, v70);
    v68 = v81;
    goto LABEL_114;
  }
  v80 = v66;
  sub_B96E90((__int64)&v92, v67, 1);
  if ( v69 == (__int64 *)&v92 )
  {
    if ( v92 )
      sub_B91220((__int64)&v92, (__int64)v92);
    goto LABEL_107;
  }
  v68 = v80;
  v70 = *(_QWORD *)(v80 + 48);
  if ( v70 )
    goto LABEL_113;
LABEL_114:
  v71 = v92;
  *(_QWORD *)(v68 + 48) = v92;
  if ( v71 )
  {
    sub_B976B0((__int64)&v92, v71, (__int64)v69);
    v26 = (__int64)v91;
    goto LABEL_108;
  }
LABEL_107:
  v26 = (__int64)v91;
LABEL_108:
  *a5 = v26;
LABEL_21:
  v27 = sub_986580((__int64)v88);
  v28 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v29 = v28;
  if ( v28 )
    sub_B4C9A0((__int64)v28, (__int64)v90, v26, a1, 3u, 0, 0, 0);
  sub_B99FD0((__int64)v29, 2u, a8);
  v30 = v29;
  result = sub_F34910(v27, v29);
  if ( !a9 )
    goto LABEL_37;
  sub_F33FD0((__int64)&v101, (unsigned __int8 *)&unk_3F298CF, (__int64 *)&v88, (__int64 *)&v90, v32);
  sub_F33FD0((__int64)&v101, (unsigned __int8 *)&unk_3F298CF, (__int64 *)&v88, (__int64 *)&v91, (__int64)&v91);
  v33 = (__int64 *)&v91;
  v34 = (__int64 *)&v90;
  if ( v25 )
  {
    sub_F33FD0((__int64)&v101, (unsigned __int8 *)&unk_3F298CF, (__int64 *)&v90, (__int64 *)&v89, (__int64)&v91);
    v33 = (__int64 *)&v91;
    if ( !v85 )
      goto LABEL_26;
  }
  else if ( !v85 )
  {
    goto LABEL_26;
  }
  sub_F33FD0((__int64)&v101, (unsigned __int8 *)&unk_3F298CF, (__int64 *)&v91, (__int64 *)&v89, (__int64)&v91);
LABEL_26:
  v35 = v96;
  if ( v99 )
  {
    v36 = HIDWORD(v97);
    v37 = (unsigned int)v102;
    v38 = &v96[HIDWORD(v97)];
    if ( v96 == v38 )
      goto LABEL_33;
LABEL_28:
    v39 = v96;
    while ( 1 )
    {
      v40 = *v39;
      v41 = v39;
      if ( (unsigned __int64)*v39 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v38 == ++v39 )
        goto LABEL_31;
    }
    if ( v39 == v38 )
    {
LABEL_31:
      if ( v99 )
      {
LABEL_74:
        v36 = HIDWORD(v97);
        goto LABEL_33;
      }
    }
    else
    {
      do
      {
        if ( HIDWORD(v102) <= (unsigned int)v37 )
        {
          sub_F35FA0((__int64)&v101, (__int64)v89, v40 & 0xFFFFFFFFFFFFFFFBLL, v37, (__int64)v33, (__int64)v34);
          v37 = (unsigned int)v102;
        }
        else
        {
          v53 = (unsigned __int8 **)&v101[16 * (unsigned int)v37];
          if ( v53 )
          {
            *v53 = v89;
            v53[1] = (unsigned __int8 *)(v40 & 0xFFFFFFFFFFFFFFFBLL);
            LODWORD(v37) = v102;
          }
          v37 = (unsigned int)(v37 + 1);
          LODWORD(v102) = v37;
        }
        v54 = v41 + 1;
        if ( v41 + 1 == v38 )
          break;
        while ( 1 )
        {
          v40 = *v54;
          v41 = v54;
          if ( (unsigned __int64)*v54 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v38 == ++v54 )
            goto LABEL_73;
        }
      }
      while ( v54 != v38 );
LABEL_73:
      v35 = v96;
      if ( v99 )
        goto LABEL_74;
    }
    v36 = (unsigned int)v97;
    goto LABEL_33;
  }
  v36 = (unsigned int)v97;
  v37 = (unsigned int)v102;
  v38 = &v96[(unsigned int)v97];
  if ( v96 != v38 )
    goto LABEL_28;
LABEL_33:
  v42 = &v35[v36];
  if ( v42 != v35 )
  {
    while ( 1 )
    {
      v43 = *v35;
      v44 = v35;
      if ( (unsigned __int64)*v35 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v42 == ++v35 )
        goto LABEL_36;
    }
    while ( v44 != v42 )
    {
      if ( HIDWORD(v102) <= (unsigned int)v37 )
      {
        sub_F35FA0((__int64)&v101, (__int64)v88, v43 | 4, v37, (__int64)v33, (__int64)v34);
        v37 = (unsigned int)v102;
      }
      else
      {
        v51 = &v101[16 * (unsigned int)v37];
        if ( v51 )
        {
          *(_QWORD *)v51 = v88;
          *((_QWORD *)v51 + 1) = v43 | 4;
          LODWORD(v37) = v102;
        }
        v37 = (unsigned int)(v37 + 1);
        LODWORD(v102) = v37;
      }
      v52 = v44 + 1;
      if ( v44 + 1 == v42 )
        break;
      while ( 1 )
      {
        v43 = *v52;
        v44 = v52;
        if ( (unsigned __int64)*v52 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v42 == ++v52 )
          goto LABEL_36;
      }
    }
  }
LABEL_36:
  v30 = v101;
  result = sub_FFB3D0(a9, v101, (unsigned int)v37);
LABEL_37:
  if ( a10 )
  {
    result = a10;
    v30 = (unsigned __int8 *)v88;
    v45 = *(_DWORD *)(a10 + 24);
    v46 = *(_QWORD *)(a10 + 8);
    if ( v45 )
    {
      v47 = v45 - 1;
      v48 = v47 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
      result = v46 + 16LL * v48;
      v49 = *(_QWORD **)result;
      if ( v88 == *(_QWORD **)result )
      {
LABEL_40:
        v50 = *(__int64 **)(result + 8);
        if ( !v50 )
          goto LABEL_44;
        if ( v25 )
        {
          sub_D4F330(*(__int64 **)(result + 8), (__int64)v90, a10);
          if ( !v85 )
            goto LABEL_43;
        }
        else if ( !v85 )
        {
LABEL_43:
          v30 = v89;
          result = (__int64)sub_D4F330(v50, (__int64)v89, a10);
          goto LABEL_44;
        }
        sub_D4F330(v50, (__int64)v91, a10);
        goto LABEL_43;
      }
      result = 1;
      while ( v49 != (_QWORD *)-4096LL )
      {
        v74 = result + 1;
        v48 = v47 & (result + v48);
        result = v46 + 16LL * v48;
        v49 = *(_QWORD **)result;
        if ( v88 == *(_QWORD **)result )
          goto LABEL_40;
        result = v74;
      }
    }
  }
LABEL_44:
  if ( !v99 )
    result = _libc_free(v96, v30);
  if ( v101 != v103 )
    return _libc_free(v101, v30);
  return result;
}
