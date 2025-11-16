// Function: sub_20F33A0
// Address: 0x20f33a0
//
__int64 __fastcall sub_20F33A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r15
  unsigned __int64 v7; // rax
  __int64 v8; // rdi
  unsigned int v9; // r11d
  __int64 *v10; // r12
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // rdx
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // r9
  unsigned __int64 v18; // r15
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rdx
  int v25; // r11d
  __int64 v26; // r8
  unsigned __int64 v27; // rcx
  unsigned int v28; // eax
  __int64 v29; // r9
  __int64 v30; // r10
  __int64 *v31; // rdx
  __int64 *v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // r8
  int v38; // r9d
  unsigned __int8 v39; // r11
  unsigned int v40; // eax
  unsigned int v41; // r15d
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 *v44; // rax
  int v45; // r15d
  __int64 v46; // rax
  unsigned __int64 v47; // rax
  __int64 v48; // r14
  __int64 v49; // rsi
  __int64 *v50; // r12
  unsigned int *v51; // rbx
  __int64 v52; // r14
  unsigned int v53; // eax
  __int64 v54; // rdx
  __int64 v55; // rsi
  unsigned __int8 *v56; // rsi
  int v57; // eax
  int v58; // r8d
  __int64 v59; // rsi
  _QWORD *v60; // rdi
  _QWORD *v61; // rcx
  __int64 *v62; // rdi
  unsigned int v63; // r9d
  __int64 *v64; // rcx
  __int64 v65; // [rsp+0h] [rbp-130h]
  unsigned int v66; // [rsp+Ch] [rbp-124h]
  __int64 v67; // [rsp+18h] [rbp-118h]
  int v68; // [rsp+18h] [rbp-118h]
  _QWORD *v69; // [rsp+20h] [rbp-110h]
  __int64 v70; // [rsp+20h] [rbp-110h]
  __int64 *v71; // [rsp+28h] [rbp-108h]
  __int64 v72; // [rsp+28h] [rbp-108h]
  __int64 v73; // [rsp+28h] [rbp-108h]
  unsigned __int8 v74; // [rsp+30h] [rbp-100h]
  __int64 v75; // [rsp+30h] [rbp-100h]
  unsigned __int8 v76; // [rsp+30h] [rbp-100h]
  unsigned __int8 v77; // [rsp+30h] [rbp-100h]
  unsigned __int8 v78; // [rsp+30h] [rbp-100h]
  unsigned __int8 v79; // [rsp+30h] [rbp-100h]
  unsigned __int8 v80; // [rsp+30h] [rbp-100h]
  unsigned __int8 v81; // [rsp+38h] [rbp-F8h]
  char v82; // [rsp+47h] [rbp-E9h]
  unsigned __int8 *v83; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v84; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v85; // [rsp+58h] [rbp-D8h]
  __int64 v86; // [rsp+60h] [rbp-D0h]
  __int64 v87; // [rsp+68h] [rbp-C8h]
  __int64 *v88; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v89; // [rsp+78h] [rbp-B8h]
  _BYTE v90[176]; // [rsp+80h] [rbp-B0h] BYREF

  v3 = a3;
  v88 = (__int64 *)v90;
  v89 = 0x800000000LL;
  v7 = a3;
  v84 = 0;
  v85 = 0;
  if ( (*(_BYTE *)(a3 + 46) & 4) != 0 )
  {
    do
      v7 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v7 + 46) & 4) != 0 );
  }
  v8 = *(_QWORD *)(a3 + 24);
  v84 = v7;
  v85 = v8 + 24;
  v86 = *(_QWORD *)(v7 + 32);
  v87 = v86 + 40LL * *(unsigned int *)(v7 + 40);
  if ( v86 == v87 )
  {
    do
    {
      v7 = *(_QWORD *)(v7 + 8);
      if ( v8 + 24 == v7 )
      {
        v84 = v8 + 24;
        goto LABEL_9;
      }
      if ( (*(_BYTE *)(v7 + 46) & 4) == 0 )
        break;
      v86 = *(_QWORD *)(v7 + 32);
      v87 = v86 + 40LL * *(unsigned int *)(v7 + 40);
    }
    while ( v86 == v87 );
    v84 = v7;
  }
LABEL_9:
  v9 = sub_1E13870(&v84, *(_DWORD *)(a2 + 112), (__int64)&v88);
  v82 = BYTE2(v9);
  if ( !(_BYTE)v9 )
    goto LABEL_10;
  v12 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL);
  if ( (*(_BYTE *)(a3 + 46) & 4) != 0 )
  {
    do
      v3 = *(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v3 + 46) & 4) != 0 );
  }
  v13 = *(_QWORD *)(v12 + 368);
  v14 = *(unsigned int *)(v12 + 384);
  if ( !(_DWORD)v14 )
    goto LABEL_32;
  v15 = (v14 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v16 = (__int64 *)(v13 + 16LL * v15);
  v17 = *v16;
  if ( *v16 != v3 )
  {
    v57 = 1;
    while ( v17 != -8 )
    {
      v58 = v57 + 1;
      v15 = (v14 - 1) & (v57 + v15);
      v16 = (__int64 *)(v13 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == v3 )
        goto LABEL_19;
      v57 = v58;
    }
LABEL_32:
    v16 = (__int64 *)(v13 + 16 * v14);
  }
LABEL_19:
  v74 = v9;
  v18 = v16[1] & 0xFFFFFFFFFFFFFFF8LL;
  v19 = (__int64 *)sub_1DB3C70((__int64 *)a2, v18);
  v9 = v74;
  if ( v19 == (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
    || (*(_DWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v19 >> 1) & 3) > *(_DWORD *)(v18 + 24)
    || (v75 = v19[2]) == 0 )
  {
    v20 = *(unsigned int *)(a3 + 40);
    if ( (_DWORD)v20 )
    {
      v21 = 5 * v20;
      v22 = 0;
      v23 = 8 * v21;
      do
      {
        v24 = v22 + *(_QWORD *)(a3 + 32);
        if ( !*(_BYTE *)v24 && (*(_BYTE *)(v24 + 3) & 0x10) == 0 && *(_DWORD *)(a2 + 112) == *(_DWORD *)(v24 + 8) )
          *(_BYTE *)(v24 + 4) |= 1u;
        v22 += 40;
      }
      while ( v23 != v22 );
    }
    goto LABEL_10;
  }
  if ( sub_1F4DD40(a1 + 168, a3) )
  {
LABEL_31:
    v10 = v88;
    v9 = 0;
    goto LABEL_11;
  }
  v25 = *(_DWORD *)(a1 + 116);
  v26 = *(_QWORD *)(a1 + 16);
  v27 = *(unsigned int *)(v26 + 408);
  v28 = v25 & 0x7FFFFFFF;
  v29 = v25 & 0x7FFFFFFF;
  v30 = 8 * v29;
  if ( (v25 & 0x7FFFFFFFu) >= (unsigned int)v27 || (v31 = *(__int64 **)(*(_QWORD *)(v26 + 400) + 8LL * v28)) == 0 )
  {
    v53 = v28 + 1;
    if ( (unsigned int)v27 < v53 )
    {
      v73 = v53;
      if ( v53 < v27 )
      {
        *(_DWORD *)(v26 + 408) = v53;
        v54 = *(_QWORD *)(v26 + 400);
        goto LABEL_61;
      }
      if ( v53 > v27 )
      {
        if ( v53 > (unsigned __int64)*(unsigned int *)(v26 + 412) )
        {
          v65 = v25 & 0x7FFFFFFF;
          v66 = v53;
          v68 = *(_DWORD *)(a1 + 116);
          v70 = *(_QWORD *)(a1 + 16);
          sub_16CD150(v26 + 400, (const void *)(v26 + 416), v53, 8, v26, v29);
          v26 = v70;
          v29 = v65;
          v53 = v66;
          v30 = 8 * v65;
          v27 = *(unsigned int *)(v70 + 408);
          v25 = v68;
        }
        v54 = *(_QWORD *)(v26 + 400);
        v59 = *(_QWORD *)(v26 + 416);
        v60 = (_QWORD *)(v54 + 8 * v73);
        v61 = (_QWORD *)(v54 + 8 * v27);
        if ( v60 != v61 )
        {
          do
            *v61++ = v59;
          while ( v60 != v61 );
          v54 = *(_QWORD *)(v26 + 400);
        }
        *(_DWORD *)(v26 + 408) = v53;
        goto LABEL_61;
      }
    }
    v54 = *(_QWORD *)(v26 + 400);
LABEL_61:
    v67 = v29;
    v69 = (_QWORD *)v26;
    *(_QWORD *)(v54 + v30) = sub_1DBA290(v25);
    v72 = *(_QWORD *)(v69[50] + 8 * v67);
    sub_1DBB110(v69, v72);
    v31 = (__int64 *)v72;
  }
  v71 = v31;
  v32 = (__int64 *)sub_1DB3C70(v31, v18 | 2);
  if ( v32 == (__int64 *)(*v71 + 24LL * *((unsigned int *)v71 + 2))
    || (*(_DWORD *)((*v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v32 >> 1) & 3) > (*(_DWORD *)(v18 + 24) | 1u) )
  {
    v85 = 0;
    v84 = v75;
    BUG();
  }
  v33 = v32[2];
  v85 = 0;
  v84 = v75;
  if ( (*(_QWORD *)(v33 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    v34 = *(_QWORD *)((*(_QWORD *)(v33 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 16);
  else
    v34 = 0;
  v35 = *(_QWORD *)(a1 + 96);
  v85 = v34;
  v39 = sub_2100320(v35, &v84, v33, v18 | 2, 0);
  if ( !v39 )
  {
    sub_20EA7E0(a1, a2, v75, v36, v37);
    v10 = v88;
    v9 = 0;
    goto LABEL_11;
  }
  if ( v82 )
  {
    sub_20EA7E0(a1, a2, v75, v36, v37);
    goto LABEL_31;
  }
  if ( *(char *)(*(_QWORD *)(v85 + 16) + 9LL) < 0 )
  {
    v76 = v39;
    v40 = sub_20F2A10(a1, v88, (unsigned int)v89, v85, v37, v38);
    v39 = v76;
    v41 = v40;
    if ( (_BYTE)v40 )
    {
      v42 = *(_QWORD *)(a1 + 96);
      v43 = v84;
      v44 = *(__int64 **)(v42 + 160);
      if ( *(__int64 **)(v42 + 168) != v44 )
        goto LABEL_44;
      v62 = &v44[*(unsigned int *)(v42 + 180)];
      v63 = *(_DWORD *)(v42 + 180);
      if ( v44 != v62 )
      {
        v64 = 0;
        while ( v84 != *v44 )
        {
          if ( *v44 == -2 )
            v64 = v44;
          if ( v62 == ++v44 )
          {
            if ( !v64 )
              goto LABEL_94;
            *v64 = v84;
            --*(_DWORD *)(v42 + 184);
            ++*(_QWORD *)(v42 + 152);
            goto LABEL_45;
          }
        }
        goto LABEL_45;
      }
LABEL_94:
      if ( v63 < *(_DWORD *)(v42 + 176) )
      {
        *(_DWORD *)(v42 + 180) = v63 + 1;
        *v62 = v43;
        ++*(_QWORD *)(v42 + 152);
      }
      else
      {
LABEL_44:
        sub_16CCBA0(v42 + 152, v84);
      }
LABEL_45:
      v10 = v88;
      v9 = v41;
      goto LABEL_11;
    }
  }
  v77 = v39;
  v45 = sub_20FFAA0(*(_QWORD *)(a1 + 96), *(unsigned int *)(a1 + 116));
  v46 = sub_21024A0(*(_QWORD *)(a1 + 96), *(_QWORD *)(a3 + 24), a3, v45, (unsigned int)&v84, *(_QWORD *)(a1 + 80), 0);
  v9 = v77;
  v47 = v46 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v47 )
    v48 = *(_QWORD *)(v47 + 16);
  else
    v48 = 0;
  v49 = *(_QWORD *)(a3 + 64);
  v50 = (__int64 *)(v48 + 64);
  v83 = (unsigned __int8 *)v49;
  if ( !v49 )
  {
    if ( v50 == (__int64 *)&v83 )
      goto LABEL_52;
    v55 = *(_QWORD *)(v48 + 64);
    if ( !v55 )
      goto LABEL_52;
LABEL_64:
    v79 = v9;
    sub_161E7C0(v48 + 64, v55);
    v9 = v79;
    goto LABEL_65;
  }
  sub_1623A60((__int64)&v83, v49, 2);
  v9 = v77;
  if ( v50 == (__int64 *)&v83 )
  {
    if ( v83 )
    {
      sub_161E7C0(v48 + 64, (__int64)v83);
      v9 = v77;
    }
    goto LABEL_52;
  }
  v55 = *(_QWORD *)(v48 + 64);
  if ( v55 )
    goto LABEL_64;
LABEL_65:
  v56 = v83;
  *(_QWORD *)(v48 + 64) = v83;
  if ( v56 )
  {
    v80 = v9;
    sub_1623210((__int64)&v83, v56, v48 + 64);
    v9 = v80;
  }
LABEL_52:
  v51 = (unsigned int *)v88;
  v10 = &v88[2 * (unsigned int)v89];
  if ( v88 != v10 )
  {
    do
    {
      v52 = *(_QWORD *)(*(_QWORD *)v51 + 32LL) + 40LL * v51[2];
      if ( !*(_BYTE *)v52 && (*(_BYTE *)(v52 + 3) & 0x10) == 0 && *(_DWORD *)(a2 + 112) == *(_DWORD *)(v52 + 8) )
      {
        v78 = v9;
        sub_1E310D0(v52, v45);
        *(_BYTE *)(v52 + 3) |= 0x40u;
        v9 = v78;
      }
      v51 += 4;
    }
    while ( v10 != (__int64 *)v51 );
LABEL_10:
    v10 = v88;
  }
LABEL_11:
  if ( v10 != (__int64 *)v90 )
  {
    v81 = v9;
    _libc_free((unsigned __int64)v10);
    return v81;
  }
  return v9;
}
