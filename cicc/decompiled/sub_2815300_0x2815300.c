// Function: sub_2815300
// Address: 0x2815300
//
__int64 __fastcall sub_2815300(_QWORD *a1, __int64 *a2, __int64 *a3, int a4)
{
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  char v9; // cl
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // r9
  __int64 v14; // r8
  char v15; // al
  __int64 v16; // rsi
  _QWORD *v18; // rbx
  _QWORD *v19; // r12
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned int v23; // eax
  _QWORD *v24; // rbx
  _QWORD *v25; // r12
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // r8
  unsigned __int64 v39; // r9
  bool v40; // zf
  __int64 v41; // r14
  __int64 v42; // rbx
  __int64 v43; // rdx
  unsigned __int64 v44; // r15
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // rdx
  int v47; // ecx
  __int64 v48; // r8
  __int64 v49; // rdx
  __int64 v50; // rcx
  int v51; // edx
  __int64 *v52; // rcx
  __int64 v53; // rdx
  unsigned __int64 *v54; // rdx
  __int64 *v55; // rbx
  unsigned __int8 *v56; // rax
  unsigned __int8 *v57; // r15
  __int64 v58; // r13
  __int64 v59; // r10
  unsigned __int64 *v60; // rsi
  __int64 v61; // r12
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  unsigned __int64 *v70; // rdx
  __int64 *v71; // [rsp+10h] [rbp-170h]
  __int64 v72; // [rsp+10h] [rbp-170h]
  __int64 v73; // [rsp+18h] [rbp-168h]
  unsigned __int64 v74; // [rsp+18h] [rbp-168h]
  unsigned __int64 v75; // [rsp+18h] [rbp-168h]
  __int64 v76; // [rsp+20h] [rbp-160h] BYREF
  _QWORD *v77; // [rsp+28h] [rbp-158h]
  __int64 v78; // [rsp+30h] [rbp-150h]
  unsigned int v79; // [rsp+38h] [rbp-148h]
  _QWORD *v80; // [rsp+48h] [rbp-138h]
  unsigned int v81; // [rsp+58h] [rbp-128h]
  char v82; // [rsp+60h] [rbp-120h]
  __int64 *v83; // [rsp+70h] [rbp-110h] BYREF
  __int64 v84; // [rsp+78h] [rbp-108h] BYREF
  __int64 v85; // [rsp+80h] [rbp-100h] BYREF
  __int64 v86; // [rsp+88h] [rbp-F8h]
  __int64 v87; // [rsp+90h] [rbp-F0h]
  unsigned __int64 *v88; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v89; // [rsp+C8h] [rbp-B8h] BYREF
  __int64 v90; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v91; // [rsp+D8h] [rbp-A8h]
  __int64 i; // [rsp+E0h] [rbp-A0h]

  v76 = 0;
  v79 = 128;
  v7 = (_QWORD *)sub_C7D670(0x2000, 8);
  v78 = 0;
  v77 = v7;
  v89 = 2;
  v8 = v7 + 1024;
  v88 = (unsigned __int64 *)&unk_49DD7B0;
  v90 = 0;
  v91 = -4096;
  for ( i = 0; v8 != v7; v7 += 8 )
  {
    if ( v7 )
    {
      v9 = v89;
      v7[2] = 0;
      v7[3] = -4096;
      *v7 = &unk_49DD7B0;
      v7[1] = v9 & 6;
      v7[4] = i;
    }
  }
  v10 = a1[155];
  v11 = a1[152];
  v12 = a2[5];
  v13 = a1[158];
  v14 = a1[153];
  v82 = 0;
  v15 = sub_2A07DE0(v12, a4, v11, v10, v14, v13, 1, (__int64)&v76);
  *((_BYTE *)a2 + 361) = v15;
  if ( v15 )
  {
    v27 = *a2;
    *((_DWORD *)a2 + 88) += a4;
    v28 = a1[156];
    v29 = *(_QWORD *)(v27 + 72);
    *(_QWORD *)(v28 + 128) = v29;
    *(_DWORD *)(v28 + 144) = *(_DWORD *)(v29 + 92);
    sub_B29120(v28);
    v30 = sub_D4B130(a2[5]);
    v31 = a2[5];
    *a2 = v30;
    a2[1] = **(_QWORD **)(v31 + 32);
    v32 = sub_D46F00(v31);
    v33 = a2[5];
    a2[2] = v32;
    v34 = sub_D47470(v33);
    v35 = a2[5];
    a2[3] = v34;
    v36 = sub_D47930(v35);
    v40 = a2[43] == 0;
    a2[4] = v36;
    v41 = v40 ? *a3 : sub_AA5780(a2[3]);
    if ( v41 )
    {
      v88 = (unsigned __int64 *)&v90;
      v89 = 0x800000000LL;
      v83 = &v85;
      v84 = 0x800000000LL;
      v42 = *(_QWORD *)(v41 + 16);
      if ( v42 )
      {
        while ( 1 )
        {
          v43 = *(_QWORD *)(v42 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v43 - 30) <= 0xAu )
            break;
          v42 = *(_QWORD *)(v42 + 8);
          if ( !v42 )
            goto LABEL_67;
        }
        v44 = v41 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_44:
        v45 = *(_QWORD *)(v43 + 40);
        if ( a2[3] != v45 )
        {
          v46 = *(_QWORD *)(v45 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v46 == v45 + 48 )
          {
            v48 = 0;
          }
          else
          {
            if ( !v46 )
              BUG();
            v47 = *(unsigned __int8 *)(v46 - 24);
            v48 = 0;
            v49 = v46 - 24;
            if ( (unsigned int)(v47 - 30) < 0xB )
              v48 = v49;
          }
          v50 = (unsigned int)v84;
          v51 = v84;
          if ( (unsigned int)v84 >= (unsigned __int64)HIDWORD(v84) )
          {
            if ( HIDWORD(v84) < (unsigned __int64)(unsigned int)v84 + 1 )
            {
              v72 = v48;
              v75 = v45;
              sub_C8D5F0((__int64)&v83, &v85, (unsigned int)v84 + 1LL, 8u, v48, v39);
              v50 = (unsigned int)v84;
              v48 = v72;
              v45 = v75;
            }
            v83[v50] = v48;
            LODWORD(v84) = v84 + 1;
          }
          else
          {
            v52 = &v83[(unsigned int)v84];
            if ( v52 )
            {
              *v52 = v48;
              v51 = v84;
            }
            LODWORD(v84) = v51 + 1;
          }
          v53 = (unsigned int)v89;
          v38 = v44 | 4;
          v37 = (unsigned int)v89;
          if ( (unsigned int)v89 >= (unsigned __int64)HIDWORD(v89) )
          {
            v39 = (unsigned int)v89 + 1LL;
            if ( HIDWORD(v89) < v39 )
            {
              v74 = v45;
              sub_C8D5F0((__int64)&v88, &v90, (unsigned int)v89 + 1LL, 0x10u, v38, v39);
              v53 = (unsigned int)v89;
              v38 = v44 | 4;
              v45 = v74;
            }
            v70 = &v88[2 * v53];
            *v70 = v45;
            v70[1] = v38;
            LODWORD(v89) = v89 + 1;
          }
          else
          {
            v54 = &v88[2 * (unsigned int)v89];
            if ( v54 )
            {
              *v54 = v45;
              v54[1] = v38;
              LODWORD(v37) = v89;
            }
            v37 = (unsigned int)(v37 + 1);
            LODWORD(v89) = v37;
          }
        }
        while ( 1 )
        {
          v42 = *(_QWORD *)(v42 + 8);
          if ( !v42 )
            break;
          v43 = *(_QWORD *)(v42 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v43 - 30) <= 0xAu )
            goto LABEL_44;
        }
        v55 = v83;
        v71 = &v83[(unsigned int)v84];
        if ( v83 != v71 )
        {
          do
          {
            v58 = *v55;
            v59 = sub_B46EC0(*v55, 0);
            if ( v41 == v59 )
              v59 = sub_B46EC0(v58, 1u);
            v73 = v59;
            v56 = (unsigned __int8 *)sub_BD2C40(72, 1u);
            v57 = v56;
            if ( v56 )
              sub_B4C8F0((__int64)v56, v73, 1u, 0, 0);
            ++v55;
            sub_F34910(v58, v57);
          }
          while ( v71 != v55 );
        }
      }
LABEL_67:
      v60 = v88;
      v61 = (__int64)(a1 + 65);
      sub_FFB3D0(v61, v88, (unsigned int)v89, v37, v38, v39);
      sub_FFCE90(v61, (__int64)v60, v62, v63, v64, v65);
      sub_FFD870(v61, (__int64)v60, v66, v67, v68, v69);
      sub_FFBC40(v61, (__int64)v60);
      if ( v83 != &v85 )
        _libc_free((unsigned __int64)v83);
      if ( v88 != (unsigned __int64 *)&v90 )
        _libc_free((unsigned __int64)v88);
    }
  }
  if ( v82 )
  {
    v23 = v81;
    v82 = 0;
    if ( v81 )
    {
      v24 = v80;
      v25 = &v80[2 * v81];
      do
      {
        if ( *v24 != -4096 && *v24 != -8192 )
        {
          v26 = v24[1];
          if ( v26 )
            sub_B91220((__int64)(v24 + 1), v26);
        }
        v24 += 2;
      }
      while ( v25 != v24 );
      v23 = v81;
    }
    sub_C7D6A0((__int64)v80, 16LL * v23, 8);
  }
  v16 = v79;
  if ( v79 )
  {
    v18 = v77;
    v84 = 2;
    v85 = 0;
    v19 = &v77[8 * (unsigned __int64)v79];
    v86 = -4096;
    v83 = (__int64 *)&unk_49DD7B0;
    v88 = (unsigned __int64 *)&unk_49DD7B0;
    v20 = -4096;
    v87 = 0;
    v89 = 2;
    v90 = 0;
    v91 = -8192;
    i = 0;
    while ( 1 )
    {
      v21 = v18[3];
      if ( v21 != v20 )
      {
        v20 = v91;
        if ( v21 != v91 )
        {
          v22 = v18[7];
          if ( v22 != -4096 && v22 != 0 && v22 != -8192 )
          {
            sub_BD60C0(v18 + 5);
            v21 = v18[3];
          }
          v20 = v21;
        }
      }
      *v18 = &unk_49DB368;
      if ( v20 != 0 && v20 != -4096 && v20 != -8192 )
        sub_BD60C0(v18 + 1);
      v18 += 8;
      if ( v19 == v18 )
        break;
      v20 = v86;
    }
    v88 = (unsigned __int64 *)&unk_49DB368;
    if ( v91 != -4096 && v91 != 0 && v91 != -8192 )
      sub_BD60C0(&v89);
    v83 = (__int64 *)&unk_49DB368;
    if ( v86 != 0 && v86 != -4096 && v86 != -8192 )
      sub_BD60C0(&v84);
    v16 = v79;
  }
  return sub_C7D6A0((__int64)v77, v16 << 6, 8);
}
