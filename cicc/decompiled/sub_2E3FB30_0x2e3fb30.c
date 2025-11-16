// Function: sub_2E3FB30
// Address: 0x2e3fb30
//
void __fastcall sub_2E3FB30(__int64 a1, __int64 a2)
{
  _QWORD *v4; // r12
  __int64 *v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // r12
  __int64 *v11; // rcx
  __int64 *v12; // r13
  __int64 *v13; // r15
  __int64 v14; // rdi
  __int64 *v15; // rdx
  __int64 v16; // rcx
  __int64 *v17; // rax
  __int64 v18; // rdx
  char v19; // dl
  __int64 *v20; // rax
  __int64 v21; // r15
  __int64 v22; // r15
  __int64 v23; // r12
  __int64 *v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rcx
  __int64 *v27; // rax
  __int64 *v28; // rdx
  __int64 *v29; // rax
  __int64 *v30; // rax
  __int64 v31; // r12
  __int64 *v32; // rcx
  __int64 *v33; // r13
  __int64 *v34; // r15
  __int64 v35; // rdi
  __int64 *v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 *v40; // rax
  __int64 v41; // rax
  unsigned int v42; // edx
  __int64 v43; // rcx
  __int64 v44; // rax
  unsigned __int64 v45; // rsi
  __int64 v46; // r13
  __int64 v47; // rbx
  __int64 v48; // r13
  __int64 *v49; // rax
  __int64 *v50; // rdx
  __int64 *v51; // rax
  __int64 *v52; // rdx
  _BYTE *v53; // rsi
  unsigned __int64 v54; // rdi
  unsigned __int64 *v55; // rbx
  unsigned __int64 *v56; // r12
  unsigned __int64 v57; // rdi
  char v58; // dl
  __int64 *v59; // rax
  __int64 v60; // rdx
  _QWORD *v61; // rcx
  __int64 v62; // rax
  __int64 v63; // [rsp+8h] [rbp-158h]
  __int64 v64; // [rsp+18h] [rbp-148h] BYREF
  __int64 v65; // [rsp+20h] [rbp-140h] BYREF
  __int64 v66; // [rsp+28h] [rbp-138h]
  __int64 *i; // [rsp+30h] [rbp-130h]
  __int64 *v68; // [rsp+38h] [rbp-128h]
  _QWORD *v69; // [rsp+40h] [rbp-120h]
  unsigned __int64 *v70; // [rsp+48h] [rbp-118h]
  __int64 *v71; // [rsp+50h] [rbp-110h]
  __int64 *v72; // [rsp+58h] [rbp-108h]
  __int64 *v73; // [rsp+60h] [rbp-100h]
  _QWORD *v74; // [rsp+68h] [rbp-F8h]
  __int64 v75; // [rsp+70h] [rbp-F0h] BYREF
  __int64 *v76; // [rsp+78h] [rbp-E8h]
  unsigned int v77; // [rsp+80h] [rbp-E0h]
  unsigned int v78; // [rsp+84h] [rbp-DCh]
  int v79; // [rsp+88h] [rbp-D8h]
  char v80; // [rsp+8Ch] [rbp-D4h]
  __int64 v81; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v82; // [rsp+D0h] [rbp-90h] BYREF
  __int64 *v83; // [rsp+D8h] [rbp-88h]
  __int64 v84; // [rsp+E0h] [rbp-80h]
  int v85; // [rsp+E8h] [rbp-78h]
  char v86; // [rsp+ECh] [rbp-74h]
  char v87; // [rsp+F0h] [rbp-70h] BYREF

  i = 0;
  v68 = 0;
  v69 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v66 = 8;
  v65 = sub_22077B0(0x40u);
  v4 = (_QWORD *)(v65 + 24);
  v5 = (__int64 *)sub_22077B0(0x200u);
  v70 = (unsigned __int64 *)(v65 + 24);
  v76 = &v81;
  v8 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(v65 + 24) = v5;
  v68 = v5;
  v9 = *(_QWORD *)(v8 + 328);
  v69 = v5 + 64;
  v74 = v4;
  v72 = v5;
  v73 = v5 + 64;
  i = v5;
  v77 = 8;
  v79 = 0;
  v80 = 1;
  if ( v5 )
    *v5 = v9;
  v81 = v9;
  v71 = v5 + 1;
  v78 = 1;
  v75 = 1;
  v10 = *v5;
  if ( v5 == v5 + 63 )
    goto LABEL_15;
LABEL_4:
  for ( i = v5 + 1; ; i = v68 )
  {
    v11 = *(__int64 **)(v10 + 112);
    v12 = &v11[*(unsigned int *)(v10 + 120)];
    v13 = v11;
    if ( v11 != v12 )
    {
      while ( 1 )
      {
        v14 = *(_QWORD *)(a1 + 112);
        v82 = *v13;
        if ( !(unsigned int)sub_2E441D0(v14, v10, v82) )
          goto LABEL_12;
        if ( v80 )
        {
          v17 = v76;
          v15 = &v76[v78];
          if ( v76 != v15 )
          {
            while ( v82 != *v17 )
            {
              if ( v15 == ++v17 )
                goto LABEL_22;
            }
            goto LABEL_12;
          }
LABEL_22:
          if ( v78 < v77 )
          {
            ++v78;
            *v15 = v82;
            ++v75;
            goto LABEL_17;
          }
        }
        sub_C8CC70((__int64)&v75, v82, (__int64)v15, v16, v6, v7);
        if ( v19 )
        {
LABEL_17:
          v20 = v71;
          if ( v71 == v73 - 1 )
          {
            ++v13;
            sub_2E3FA50((unsigned __int64 *)&v65, &v82);
            if ( v12 == v13 )
              break;
          }
          else
          {
            if ( v71 )
            {
              *v71 = v82;
              v20 = v71;
            }
            ++v13;
            v71 = v20 + 1;
            if ( v12 == v13 )
              break;
          }
        }
        else
        {
LABEL_12:
          if ( v12 == ++v13 )
            break;
        }
      }
    }
    v5 = i;
    if ( i == v71 )
      break;
    v10 = *i;
    if ( i != v69 - 1 )
      goto LABEL_4;
LABEL_15:
    j_j___libc_free_0((unsigned __int64)v68);
    v18 = *++v70 + 512;
    v68 = (__int64 *)*v70;
    v69 = (_QWORD *)v18;
  }
  v21 = *(_QWORD *)(a1 + 128);
  v86 = 1;
  v82 = 0;
  v83 = (__int64 *)&v87;
  v22 = v21 + 320;
  v84 = 8;
  v85 = 0;
  v23 = *(_QWORD *)(v22 + 8);
  if ( v22 == v23 )
  {
LABEL_104:
    v45 = 0;
    goto LABEL_60;
  }
  do
  {
    if ( *(_DWORD *)(v23 + 120) )
      goto LABEL_28;
    if ( v80 )
    {
      v24 = v76;
      v25 = &v76[v78];
      if ( v76 == v25 )
        goto LABEL_28;
      while ( *v24 != v23 )
      {
        if ( v25 == ++v24 )
          goto LABEL_28;
      }
    }
    else if ( !sub_C8CA60((__int64)&v75, v23) )
    {
      goto LABEL_28;
    }
    v26 = (__int64)v73;
    v27 = v71;
    v28 = v73 - 1;
    if ( v71 == v73 - 1 )
    {
      v61 = v74;
      if ( v69 - i + ((v74 - v70 - 1) << 6) + v71 - v72 == 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
      if ( (unsigned __int64)(v66 - (((__int64)v74 - v65) >> 3)) <= 1 )
      {
        sub_2E3F8D0((unsigned __int64 *)&v65, 1u, 0);
        v61 = v74;
      }
      v63 = (__int64)v61;
      v62 = sub_22077B0(0x200u);
      v26 = v63;
      *(_QWORD *)(v63 + 8) = v62;
      if ( v71 )
        *v71 = v23;
      v28 = (__int64 *)(*++v74 + 512LL);
      v72 = (__int64 *)*v74;
      v73 = v28;
      v71 = v72;
    }
    else
    {
      if ( v71 )
      {
        *v71 = v23;
        v27 = v71;
      }
      v71 = v27 + 1;
    }
    if ( !v86 )
      goto LABEL_103;
    v29 = v83;
    v26 = HIDWORD(v84);
    v28 = &v83[HIDWORD(v84)];
    if ( v83 == v28 )
    {
LABEL_43:
      if ( HIDWORD(v84) < (unsigned int)v84 )
      {
        ++HIDWORD(v84);
        *v28 = v23;
        ++v82;
        goto LABEL_28;
      }
LABEL_103:
      sub_C8CC70((__int64)&v82, v23, (__int64)v28, v26, v6, v7);
      goto LABEL_28;
    }
    while ( *v29 != v23 )
    {
      if ( v28 == ++v29 )
        goto LABEL_43;
    }
LABEL_28:
    v23 = *(_QWORD *)(v23 + 8);
  }
  while ( v22 != v23 );
  v30 = i;
  if ( v71 != i )
  {
    while ( 1 )
    {
      v31 = *v30;
      if ( v30 == v69 - 1 )
      {
        j_j___libc_free_0((unsigned __int64)v68);
        v60 = *++v70 + 512;
        v68 = (__int64 *)*v70;
        v69 = (_QWORD *)v60;
        i = v68;
      }
      else
      {
        i = v30 + 1;
      }
      v32 = *(__int64 **)(v31 + 64);
      v33 = &v32[*(unsigned int *)(v31 + 72)];
      v34 = v32;
      if ( v32 != v33 )
        break;
LABEL_56:
      v30 = i;
      if ( v71 == i )
        goto LABEL_57;
    }
    while ( 1 )
    {
      v35 = *(_QWORD *)(a1 + 112);
      v64 = *v34;
      if ( !(unsigned int)sub_2E441D0(v35, v64, v31) )
        goto LABEL_55;
      if ( !v86 )
        goto LABEL_86;
      v40 = v83;
      v36 = &v83[HIDWORD(v84)];
      if ( v83 != v36 )
      {
        while ( v64 != *v40 )
        {
          if ( v36 == ++v40 )
            goto LABEL_92;
        }
LABEL_55:
        if ( v33 == ++v34 )
          goto LABEL_56;
        continue;
      }
LABEL_92:
      if ( HIDWORD(v84) < (unsigned int)v84 )
      {
        ++HIDWORD(v84);
        *v36 = v64;
        ++v82;
      }
      else
      {
LABEL_86:
        sub_C8CC70((__int64)&v82, v64, (__int64)v36, v37, v38, v39);
        if ( !v58 )
          goto LABEL_55;
      }
      v59 = v71;
      if ( v71 == v73 - 1 )
      {
        ++v34;
        sub_2E3FA50((unsigned __int64 *)&v65, &v64);
        if ( v33 == v34 )
          goto LABEL_56;
      }
      else
      {
        if ( v71 )
        {
          *v71 = v64;
          v59 = v71;
        }
        ++v34;
        v71 = v59 + 1;
        if ( v33 == v34 )
          goto LABEL_56;
      }
    }
  }
LABEL_57:
  v41 = *(_QWORD *)(a1 + 128);
  v42 = 0;
  v43 = v41 + 320;
  v44 = *(_QWORD *)(v41 + 328);
  if ( v43 == v44 )
    goto LABEL_104;
  do
  {
    v44 = *(_QWORD *)(v44 + 8);
    ++v42;
  }
  while ( v43 != v44 );
  v45 = v42;
LABEL_60:
  sub_2E3A980(a2, v45);
  v46 = *(_QWORD *)(a1 + 128);
  v47 = *(_QWORD *)(v46 + 328);
  v48 = v46 + 320;
  if ( v48 != v47 )
  {
    while ( 2 )
    {
      if ( v80 )
      {
        v49 = v76;
        v50 = &v76[v78];
        if ( v76 == v50 )
          goto LABEL_75;
        while ( *v49 != v47 )
        {
          if ( v50 == ++v49 )
            goto LABEL_75;
        }
      }
      else if ( !sub_C8CA60((__int64)&v75, v47) )
      {
        goto LABEL_75;
      }
      if ( v86 )
      {
        v51 = v83;
        v52 = &v83[HIDWORD(v84)];
        if ( v83 == v52 )
          goto LABEL_75;
        while ( *v51 != v47 )
        {
          if ( v52 == ++v51 )
            goto LABEL_75;
        }
      }
      else if ( !sub_C8CA60((__int64)&v82, v47) )
      {
        goto LABEL_75;
      }
      v64 = v47;
      v53 = *(_BYTE **)(a2 + 8);
      if ( v53 == *(_BYTE **)(a2 + 16) )
      {
        sub_2E3CE90(a2, v53, &v64);
      }
      else
      {
        if ( v53 )
        {
          *(_QWORD *)v53 = v47;
          v53 = *(_BYTE **)(a2 + 8);
        }
        *(_QWORD *)(a2 + 8) = v53 + 8;
      }
LABEL_75:
      v47 = *(_QWORD *)(v47 + 8);
      if ( v48 == v47 )
        break;
      continue;
    }
  }
  if ( !v86 )
    _libc_free((unsigned __int64)v83);
  if ( !v80 )
    _libc_free((unsigned __int64)v76);
  v54 = v65;
  if ( v65 )
  {
    v55 = v70;
    v56 = v74 + 1;
    if ( v74 + 1 > v70 )
    {
      do
      {
        v57 = *v55++;
        j_j___libc_free_0(v57);
      }
      while ( v56 > v55 );
      v54 = v65;
    }
    j_j___libc_free_0(v54);
  }
}
