// Function: sub_1B6D8B0
// Address: 0x1b6d8b0
//
__int64 __fastcall sub_1B6D8B0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // r15
  char v15; // al
  _BYTE *v16; // rsi
  _QWORD **v17; // rdx
  _BYTE *v18; // rax
  _QWORD *v19; // rdx
  unsigned int v20; // r12d
  __int64 *v22; // rsi
  __int64 v23; // r12
  _QWORD *v24; // rax
  _QWORD *v25; // rbx
  __int64 v26; // r12
  _QWORD *v27; // rdi
  __int64 *v28; // r12
  __int64 v29; // r14
  _QWORD *v30; // rbx
  unsigned __int64 *v31; // rcx
  unsigned __int64 v32; // rdx
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 v35; // r15
  _QWORD *v36; // rdi
  __int64 v37; // r14
  _QWORD *v38; // rax
  _QWORD *v39; // r12
  __int64 v40; // rax
  __int64 v41; // r15
  _QWORD *v42; // rax
  __int64 v43; // r14
  __int64 *v44; // rbx
  __int64 *v45; // rax
  __int64 v46; // rsi
  int v47; // eax
  __int64 v48; // rax
  int v49; // ecx
  __int64 v50; // rcx
  __int64 *v51; // rax
  __int64 v52; // rdi
  unsigned __int64 v53; // rcx
  __int64 v54; // rcx
  __int64 v55; // rcx
  __int64 v56; // rsi
  __int64 *v57; // rdi
  unsigned __int64 v58; // rsi
  __int64 v59; // rcx
  __int64 v60; // rdx
  double v61; // xmm4_8
  double v62; // xmm5_8
  _QWORD *v63; // rdi
  __int64 v64; // r15
  unsigned __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // r15
  __int64 v71; // rbx
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rax
  __int64 v75; // r15
  _QWORD *v76; // rdi
  __int64 *v78; // [rsp+8h] [rbp-A8h]
  _QWORD *v79; // [rsp+10h] [rbp-A0h]
  __int64 *i; // [rsp+18h] [rbp-98h]
  _BYTE *v81; // [rsp+20h] [rbp-90h] BYREF
  _BYTE *v82; // [rsp+28h] [rbp-88h]
  _BYTE *v83; // [rsp+30h] [rbp-80h]
  __int64 *v84; // [rsp+40h] [rbp-70h] BYREF
  __int64 *v85; // [rsp+48h] [rbp-68h]
  __int64 *v86; // [rsp+50h] [rbp-60h]
  __int64 v87[2]; // [rsp+60h] [rbp-50h] BYREF
  char v88; // [rsp+70h] [rbp-40h]
  char v89; // [rsp+71h] [rbp-3Fh]

  v11 = a2 + 72;
  v12 = *(_QWORD *)(a2 + 80);
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  if ( v12 == a2 + 72 )
  {
    v17 = 0;
    v18 = 0;
    goto LABEL_78;
  }
  do
  {
    while ( 1 )
    {
      v13 = v12 - 24;
      if ( !v12 )
        v13 = 0;
      v14 = v13;
      v15 = *(_BYTE *)(sub_157EBA0(v13) + 16);
      if ( v15 != 25 )
      {
        if ( v15 == 31 )
        {
          v87[0] = v14;
          v22 = v85;
          if ( v85 == v86 )
          {
            sub_15D0700((__int64)&v84, v85, v87);
          }
          else
          {
            if ( v85 )
            {
              *v85 = v14;
              v22 = v85;
            }
            v85 = v22 + 1;
          }
        }
        goto LABEL_4;
      }
      v87[0] = v14;
      v16 = v82;
      if ( v82 != v83 )
        break;
      sub_15D0700((__int64)&v81, v82, v87);
LABEL_4:
      v12 = *(_QWORD *)(v12 + 8);
      if ( v12 == v11 )
        goto LABEL_12;
    }
    if ( v82 )
    {
      *(_QWORD *)v82 = v14;
      v16 = v82;
    }
    v82 = v16 + 8;
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( v12 != v11 );
LABEL_12:
  if ( v85 == v84 )
  {
    v18 = v82;
    v17 = (_QWORD **)v81;
LABEL_78:
    *(_QWORD *)(a1 + 176) = 0;
LABEL_15:
    if ( v18 == (_BYTE *)v17 )
      goto LABEL_37;
    goto LABEL_16;
  }
  if ( (char *)v85 - (char *)v84 == 8 )
  {
    v17 = (_QWORD **)v81;
    *(_QWORD *)(a1 + 176) = *v84;
    v18 = v82;
    goto LABEL_15;
  }
  v89 = 1;
  v87[0] = (__int64)"UnifiedUnreachableBlock";
  v88 = 3;
  v23 = sub_15E0530(a2);
  v24 = (_QWORD *)sub_22077B0(64);
  v25 = v24;
  if ( v24 )
    sub_157FB60(v24, v23, (__int64)v87, a2, 0);
  *(_QWORD *)(a1 + 176) = v25;
  v26 = sub_15E0530(a2);
  v27 = sub_1648A60(56, 0);
  if ( v27 )
    sub_15F82E0((__int64)v27, v26, *(_QWORD *)(a1 + 176));
  v28 = v84;
  for ( i = v85; i != v28; ++v28 )
  {
    v29 = *v28;
    v30 = (_QWORD *)(*(_QWORD *)(*v28 + 40) & 0xFFFFFFFFFFFFFFF8LL);
    sub_157EA20(*v28 + 40, (__int64)(v30 - 3));
    v31 = (unsigned __int64 *)v30[1];
    v32 = *v30 & 0xFFFFFFFFFFFFFFF8LL;
    *v31 = v32 | *v31 & 7;
    *(_QWORD *)(v32 + 8) = v31;
    *v30 &= 7uLL;
    v30[1] = 0;
    sub_164BEC0((__int64)(v30 - 3), (__int64)(v30 - 3), v32, (__int64)v31, a3, a4, a5, a6, v33, v34, a9, a10);
    v35 = *(_QWORD *)(a1 + 176);
    v36 = sub_1648A60(56, 1u);
    if ( v36 )
      sub_15F8590((__int64)v36, v35, v29);
  }
  v18 = v82;
  v17 = (_QWORD **)v81;
  if ( v82 == v81 )
  {
LABEL_37:
    *(_QWORD *)(a1 + 160) = 0;
    v20 = 0;
    goto LABEL_19;
  }
LABEL_16:
  if ( v18 - (_BYTE *)v17 == 8 )
  {
    v19 = *v17;
    v20 = *(unsigned __int8 *)(a1 + 184);
    *(_QWORD *)(a1 + 160) = v19;
    if ( (_BYTE)v20 )
      v20 = sub_1B6D870(a1, a2, v19);
  }
  else
  {
    v89 = 1;
    v87[0] = (__int64)"UnifiedReturnBlock";
    v88 = 3;
    v37 = sub_15E0530(a2);
    v38 = (_QWORD *)sub_22077B0(64);
    v39 = v38;
    if ( v38 )
      sub_157FB60(v38, v37, (__int64)v87, a2, 0);
    v40 = *(_QWORD *)(a2 + 24);
    if ( *(_BYTE *)(**(_QWORD **)(v40 + 16) + 8LL) )
    {
      v89 = 1;
      v87[0] = (__int64)"UnifiedRetVal";
      v88 = 3;
      v70 = **(_QWORD **)(v40 + 16);
      v71 = (v82 - v81) >> 3;
      v72 = sub_1648B60(64);
      v43 = v72;
      if ( v72 )
      {
        sub_15F1EA0(v72, v70, 53, 0, 0, 0);
        *(_DWORD *)(v43 + 56) = v71;
        sub_164B780(v43, v87);
        sub_1648880(v43, *(_DWORD *)(v43 + 56), 1);
      }
      sub_157E9D0((__int64)(v39 + 5), v43);
      v73 = v39[5];
      v74 = *(_QWORD *)(v43 + 24);
      *(_QWORD *)(v43 + 32) = v39 + 5;
      v73 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v43 + 24) = v73 | v74 & 7;
      *(_QWORD *)(v73 + 8) = v43 + 24;
      v39[5] = v39[5] & 7LL | (v43 + 24);
      v75 = sub_15E0530(a2);
      v76 = sub_1648A60(56, 1u);
      if ( v76 )
        sub_15F7090((__int64)v76, v75, v43, (__int64)v39);
    }
    else
    {
      v41 = sub_15E0530(a2);
      v42 = sub_1648A60(56, 0);
      v43 = (__int64)v42;
      if ( v42 )
      {
        sub_15F7090((__int64)v42, v41, 0, (__int64)v39);
        v43 = 0;
      }
    }
    v44 = (__int64 *)v81;
    v78 = (__int64 *)v82;
    if ( v82 != v81 )
    {
      do
      {
        v64 = *v44;
        if ( v43 )
        {
          v65 = sub_157EBA0(*v44);
          if ( (*(_BYTE *)(v65 + 23) & 0x40) != 0 )
          {
            v45 = *(__int64 **)(v65 - 8);
          }
          else
          {
            v67 = 24LL * (*(_DWORD *)(v65 + 20) & 0xFFFFFFF);
            v45 = (__int64 *)(v65 - v67);
          }
          v46 = *v45;
          v47 = *(_DWORD *)(v43 + 20) & 0xFFFFFFF;
          if ( v47 == *(_DWORD *)(v43 + 56) )
          {
            sub_15F55D0(v43, v46, v66, v67, v68, v69);
            v47 = *(_DWORD *)(v43 + 20) & 0xFFFFFFF;
          }
          v48 = (v47 + 1) & 0xFFFFFFF;
          v49 = v48 | *(_DWORD *)(v43 + 20) & 0xF0000000;
          *(_DWORD *)(v43 + 20) = v49;
          if ( (v49 & 0x40000000) != 0 )
            v50 = *(_QWORD *)(v43 - 8);
          else
            v50 = v43 - 24 * v48;
          v51 = (__int64 *)(v50 + 24LL * (unsigned int)(v48 - 1));
          if ( *v51 )
          {
            v52 = v51[1];
            v53 = v51[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v53 = v52;
            if ( v52 )
              *(_QWORD *)(v52 + 16) = *(_QWORD *)(v52 + 16) & 3LL | v53;
          }
          *v51 = v46;
          if ( v46 )
          {
            v54 = *(_QWORD *)(v46 + 8);
            v51[1] = v54;
            if ( v54 )
              *(_QWORD *)(v54 + 16) = (unsigned __int64)(v51 + 1) | *(_QWORD *)(v54 + 16) & 3LL;
            v51[2] = (v46 + 8) | v51[2] & 3;
            *(_QWORD *)(v46 + 8) = v51;
          }
          v55 = *(_DWORD *)(v43 + 20) & 0xFFFFFFF;
          if ( (*(_BYTE *)(v43 + 23) & 0x40) != 0 )
            v56 = *(_QWORD *)(v43 - 8);
          else
            v56 = v43 - 24 * v55;
          *(_QWORD *)(v56 + 8LL * (unsigned int)(v55 - 1) + 24LL * *(unsigned int *)(v43 + 56) + 8) = v64;
        }
        v79 = (_QWORD *)(*(_QWORD *)(v64 + 40) & 0xFFFFFFFFFFFFFFF8LL);
        sub_157EA20(v64 + 40, (__int64)(v79 - 3));
        v57 = (__int64 *)v79[1];
        v58 = *v79 & 0xFFFFFFFFFFFFFFF8LL;
        v59 = v58 | *v57 & 7;
        *v57 = v59;
        *(_QWORD *)(v58 + 8) = v57;
        *v79 &= 7uLL;
        v79[1] = 0;
        sub_164BEC0((__int64)(v79 - 3), v58, v60, v59, a3, a4, a5, a6, v61, v62, a9, a10);
        v63 = sub_1648A60(56, 1u);
        if ( v63 )
          sub_15F8590((__int64)v63, (__int64)v39, v64);
        ++v44;
      }
      while ( v78 != v44 );
    }
    if ( *(_BYTE *)(a1 + 184) )
      sub_1B6D870(a1, a2, v39);
    *(_QWORD *)(a1 + 160) = v39;
    v20 = 1;
  }
LABEL_19:
  if ( v84 )
    j_j___libc_free_0(v84, (char *)v86 - (char *)v84);
  if ( v81 )
    j_j___libc_free_0(v81, v83 - v81);
  return v20;
}
