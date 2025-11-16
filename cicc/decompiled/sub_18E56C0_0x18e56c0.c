// Function: sub_18E56C0
// Address: 0x18e56c0
//
__int64 __fastcall sub_18E56C0(
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
  __int64 v10; // rbx
  __int64 j; // r14
  __int64 *v12; // r12
  unsigned __int8 v13; // r13
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rax
  _QWORD *v17; // rax
  _QWORD *v18; // r12
  __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  int v22; // r15d
  __int64 v23; // rbx
  __int64 *v24; // rax
  __int64 *v25; // r12
  __int64 v26; // rdi
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // r8d
  int v30; // r9d
  __int64 v31; // rax
  int v32; // eax
  _BYTE *v33; // rdi
  __int64 i; // r14
  __int64 *v35; // rax
  __int64 *v36; // r15
  __int64 **v37; // rax
  __int64 **v38; // rbx
  unsigned __int64 v39; // rax
  int v40; // r8d
  int v41; // r9d
  unsigned int v42; // ebx
  __int64 v43; // rax
  _QWORD **v44; // rbx
  _BYTE *v45; // r12
  _QWORD *v46; // rdi
  __int64 v48; // rax
  double v49; // xmm4_8
  double v50; // xmm5_8
  __int64 v51; // rdx
  __int64 *v52; // rdi
  __int64 *v53; // rcx
  __int64 **v54; // rdx
  int v55; // [rsp+18h] [rbp-5C8h]
  __int64 v56; // [rsp+18h] [rbp-5C8h]
  __int64 v57; // [rsp+20h] [rbp-5C0h]
  unsigned __int8 v58; // [rsp+30h] [rbp-5B0h]
  __int64 v59; // [rsp+38h] [rbp-5A8h]
  __int64 v60; // [rsp+48h] [rbp-598h]
  __int64 v61; // [rsp+50h] [rbp-590h] BYREF
  unsigned int v62; // [rsp+58h] [rbp-588h]
  _BYTE *v63; // [rsp+60h] [rbp-580h] BYREF
  __int64 v64; // [rsp+68h] [rbp-578h]
  _BYTE v65[128]; // [rsp+70h] [rbp-570h] BYREF
  __int64 v66; // [rsp+F0h] [rbp-4F0h] BYREF
  __int64 *v67; // [rsp+F8h] [rbp-4E8h]
  __int64 *v68; // [rsp+100h] [rbp-4E0h]
  __int64 v69; // [rsp+108h] [rbp-4D8h]
  int v70; // [rsp+110h] [rbp-4D0h]
  _BYTE v71[136]; // [rsp+118h] [rbp-4C8h] BYREF
  _BYTE *v72; // [rsp+1A0h] [rbp-440h] BYREF
  __int64 v73; // [rsp+1A8h] [rbp-438h]
  _BYTE v74[1072]; // [rsp+1B0h] [rbp-430h] BYREF

  v10 = *(_QWORD *)(a1 + 80);
  v72 = v74;
  v73 = 0x8000000000LL;
  v59 = a1 + 72;
  if ( a1 + 72 == v10 )
  {
    j = 0;
  }
  else
  {
    if ( !v10 )
      BUG();
    while ( 1 )
    {
      j = *(_QWORD *)(v10 + 24);
      if ( j != v10 + 16 )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( a1 + 72 == v10 )
        break;
      if ( !v10 )
        BUG();
    }
  }
  v58 = 0;
  while ( v10 != v59 )
  {
    v12 = (__int64 *)(j - 24);
    if ( !j )
      v12 = 0;
    v60 = (__int64)v12;
    if ( (unsigned __int8)sub_15F3040((__int64)v12) || sub_15F3330((__int64)v12) )
    {
      if ( !v12[1] )
        goto LABEL_25;
      if ( *(_BYTE *)(*v12 + 8) != 11 )
        goto LABEL_14;
    }
    else if ( *(_BYTE *)(*v12 + 8) != 11 )
    {
      goto LABEL_14;
    }
    sub_13A29C0((__int64)&v66, a2, v12);
    v22 = (int)v67;
    if ( (unsigned int)v67 <= 0x40 )
    {
      if ( v66 )
        goto LABEL_14;
    }
    else
    {
      if ( v22 != (unsigned int)sub_16A57B0((__int64)&v66) )
      {
        if ( v66 )
          j_j___libc_free_0_0(v66);
        goto LABEL_14;
      }
      if ( v66 )
        j_j___libc_free_0_0(v66);
    }
    v63 = v65;
    v64 = 0x1000000000LL;
    if ( !v12[1] )
      goto LABEL_93;
    v57 = v10;
    v23 = v12[1];
    do
    {
      while ( 1 )
      {
        v24 = sub_1648700(v23);
        v25 = v24;
        if ( *((_BYTE *)v24 + 16) <= 0x17u )
          goto LABEL_43;
        v26 = *v24;
        v27 = *(unsigned __int8 *)(*v24 + 8);
        if ( (unsigned __int8)v27 > 0xFu || (v28 = 35454, !_bittest64(&v28, v27)) )
        {
          if ( (unsigned int)(v27 - 13) > 1 && (_DWORD)v27 != 16 || !sub_16435F0(v26, 0) )
            goto LABEL_43;
        }
        sub_13A29C0((__int64)&v66, a2, v25);
        if ( (unsigned int)v67 > 0x40 )
          break;
        if ( v66 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v67) )
        {
          v31 = (unsigned int)v64;
          if ( (unsigned int)v64 < HIDWORD(v64) )
            goto LABEL_52;
LABEL_124:
          sub_16CD150((__int64)&v63, v65, 0, 8, v29, v30);
          v31 = (unsigned int)v64;
          goto LABEL_52;
        }
LABEL_43:
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          goto LABEL_53;
      }
      v55 = (int)v67;
      if ( v55 == (unsigned int)sub_16A58F0((__int64)&v66) )
      {
        if ( v66 )
          j_j___libc_free_0_0(v66);
        goto LABEL_43;
      }
      if ( v66 )
        j_j___libc_free_0_0(v66);
      v31 = (unsigned int)v64;
      if ( (unsigned int)v64 >= HIDWORD(v64) )
        goto LABEL_124;
LABEL_52:
      *(_QWORD *)&v63[8 * v31] = v25;
      LODWORD(v64) = v64 + 1;
      v23 = *(_QWORD *)(v23 + 8);
    }
    while ( v23 );
LABEL_53:
    v32 = v64;
    v66 = 0;
    v67 = (__int64 *)v71;
    v10 = v57;
    v68 = (__int64 *)v71;
    v33 = v63;
    v69 = 16;
    v70 = 0;
    if ( !(_DWORD)v64 )
      goto LABEL_91;
    v56 = j;
    while ( 2 )
    {
      i = *(_QWORD *)&v33[8 * v32 - 8];
      LODWORD(v64) = v32 - 1;
      sub_15F2390(i);
      v35 = v67;
      if ( v68 == v67 )
      {
        v52 = &v67[HIDWORD(v69)];
        if ( v67 == v52 )
        {
LABEL_120:
          if ( HIDWORD(v69) >= (unsigned int)v69 )
            goto LABEL_56;
          ++HIDWORD(v69);
          *v52 = i;
          ++v66;
        }
        else
        {
          v53 = 0;
          while ( i != *v35 )
          {
            if ( *v35 == -2 )
              v53 = v35;
            if ( v52 == ++v35 )
            {
              if ( !v53 )
                goto LABEL_120;
              *v53 = i;
              --v70;
              ++v66;
              break;
            }
          }
        }
      }
      else
      {
LABEL_56:
        sub_16CCBA0((__int64)&v66, i);
      }
LABEL_57:
      for ( i = *(_QWORD *)(i + 8); i; i = *(_QWORD *)(i + 8) )
      {
        v36 = sub_1648700(i);
        if ( *((_BYTE *)v36 + 16) <= 0x17u )
          goto LABEL_57;
        v37 = (__int64 **)v67;
        if ( v68 == v67 )
        {
          v38 = (__int64 **)&v67[HIDWORD(v69)];
          if ( v67 == (__int64 *)v38 )
          {
            v54 = (__int64 **)v67;
          }
          else
          {
            do
            {
              if ( v36 == *v37 )
                break;
              ++v37;
            }
            while ( v38 != v37 );
            v54 = (__int64 **)&v67[HIDWORD(v69)];
          }
        }
        else
        {
          v38 = (__int64 **)&v68[(unsigned int)v69];
          v37 = (__int64 **)sub_16CC9F0((__int64)&v66, (__int64)v36);
          if ( v36 == *v37 )
          {
            if ( v68 == v67 )
              v54 = (__int64 **)&v68[HIDWORD(v69)];
            else
              v54 = (__int64 **)&v68[(unsigned int)v69];
          }
          else
          {
            if ( v68 != v67 )
            {
              v37 = (__int64 **)&v68[(unsigned int)v69];
              goto LABEL_63;
            }
            v37 = (__int64 **)&v68[HIDWORD(v69)];
            v54 = v37;
          }
        }
        while ( v54 != v37 && (unsigned __int64)*v37 >= 0xFFFFFFFFFFFFFFFELL )
          ++v37;
LABEL_63:
        if ( v37 != v38 )
          goto LABEL_57;
        v39 = *(unsigned __int8 *)(*v36 + 8);
        if ( (unsigned __int8)v39 > 0xFu || (v51 = 35454, !_bittest64(&v51, v39)) )
        {
          if ( (unsigned int)(v39 - 13) > 1 && (_DWORD)v39 != 16 || !sub_16435F0(*v36, 0) )
            goto LABEL_57;
        }
        sub_13A29C0((__int64)&v61, a2, v36);
        v42 = v62;
        if ( v62 <= 0x40 )
        {
          if ( v61 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v62) )
            goto LABEL_57;
        }
        else
        {
          if ( v42 == (unsigned int)sub_16A58F0((__int64)&v61) )
          {
            if ( v61 )
              j_j___libc_free_0_0(v61);
            goto LABEL_57;
          }
          if ( v61 )
            j_j___libc_free_0_0(v61);
        }
        v43 = (unsigned int)v64;
        if ( (unsigned int)v64 >= HIDWORD(v64) )
        {
          sub_16CD150((__int64)&v63, v65, 0, 8, v40, v41);
          v43 = (unsigned int)v64;
        }
        *(_QWORD *)&v63[8 * v43] = v36;
        LODWORD(v64) = v64 + 1;
      }
      v32 = v64;
      if ( (_DWORD)v64 )
      {
        v33 = v63;
        continue;
      }
      break;
    }
    v10 = v57;
    j = v56;
    if ( v68 != v67 )
      _libc_free((unsigned __int64)v68);
    v33 = v63;
LABEL_91:
    if ( v33 != v65 )
      _libc_free((unsigned __int64)v33);
LABEL_93:
    v48 = sub_15A0680(*(_QWORD *)v60, 0, 0);
    sub_164D170(v60, v48, a3, a4, a5, a6, v49, v50, a9, a10);
    v58 = 1;
LABEL_14:
    v13 = sub_13A2D40(a2, v60);
    if ( v13 )
    {
      sub_1AEAA40(v60);
      v16 = (unsigned int)v73;
      if ( (unsigned int)v73 >= HIDWORD(v73) )
      {
        sub_16CD150((__int64)&v72, v74, 0, 8, v14, v15);
        v16 = (unsigned int)v73;
      }
      *(_QWORD *)&v72[8 * v16] = v60;
      LODWORD(v73) = v73 + 1;
      if ( (*(_BYTE *)(v60 + 23) & 0x40) != 0 )
      {
        v17 = *(_QWORD **)(v60 - 8);
        v18 = &v17[3 * (*(_DWORD *)(v60 + 20) & 0xFFFFFFF)];
      }
      else
      {
        v18 = (_QWORD *)v60;
        v17 = (_QWORD *)(v60 - 24LL * (*(_DWORD *)(v60 + 20) & 0xFFFFFFF));
      }
      for ( ; v18 != v17; v17 += 3 )
      {
        if ( *v17 )
        {
          v19 = v17[1];
          v20 = v17[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v20 = v19;
          if ( v19 )
            *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
        }
        *v17 = 0;
      }
      v58 = v13;
    }
LABEL_25:
    for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v10 + 24) )
    {
      v21 = v10 - 24;
      if ( !v10 )
        v21 = 0;
      if ( j != v21 + 40 )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( v59 == v10 )
        break;
      if ( !v10 )
        BUG();
    }
  }
  v44 = (_QWORD **)v72;
  v45 = &v72[8 * (unsigned int)v73];
  if ( v72 != v45 )
  {
    do
    {
      v46 = *v44++;
      sub_15F20C0(v46);
    }
    while ( v45 != (_BYTE *)v44 );
    v45 = v72;
  }
  if ( v45 != v74 )
    _libc_free((unsigned __int64)v45);
  return v58;
}
