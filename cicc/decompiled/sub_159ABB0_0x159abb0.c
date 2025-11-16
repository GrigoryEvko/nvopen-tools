// Function: sub_159ABB0
// Address: 0x159abb0
//
__int64 __fastcall sub_159ABB0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // r12
  __int64 **v5; // r15
  __int64 v6; // rdx
  __int64 v7; // rcx
  bool v8; // al
  bool v9; // cl
  __int16 **v10; // rdi
  __int64 v11; // rcx
  bool v12; // r8
  __int64 result; // rax
  char v14; // al
  __int64 *v15; // r15
  __int64 *v16; // r14
  __int64 v17; // rax
  _QWORD *v18; // r13
  __int64 v19; // rax
  unsigned int v20; // eax
  char *v21; // r14
  size_t v22; // r13
  __int64 v23; // rax
  unsigned int v24; // eax
  char v25; // al
  __int64 *v26; // r14
  __int64 *v27; // r13
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 v30; // rax
  char *v31; // r14
  __int64 v32; // r13
  __int64 v33; // rax
  __int64 *v34; // r15
  __int64 *v35; // r14
  __int64 v36; // rax
  _QWORD *v37; // r13
  __int64 v38; // rax
  unsigned int v39; // eax
  char *v40; // r14
  __int64 v41; // r13
  __int64 v42; // rax
  __int64 *v43; // r15
  __int64 *v44; // r14
  __int64 v45; // rax
  _QWORD *v46; // r13
  __int64 v47; // rax
  char *v48; // r14
  __int64 v49; // r13
  __int64 v50; // rax
  __int64 *v51; // r15
  __int64 *v52; // r13
  __int64 v53; // rax
  _QWORD *v54; // r14
  __int64 v55; // rax
  char *v56; // r14
  __int64 v57; // r13
  __int64 v58; // rax
  __int64 *v59; // r14
  __int64 *v60; // r13
  __int64 v61; // rax
  __int16 v62; // r8
  __int64 v63; // rax
  char *v64; // r14
  __int64 v65; // r13
  __int64 v66; // rax
  __int64 *v67; // r15
  __int64 *v68; // r13
  __int64 v69; // rax
  int v70; // r8d
  __int64 v71; // rax
  char *v72; // r14
  __int64 v73; // r13
  __int64 v74; // rax
  int v75; // eax
  int v76; // eax
  int v77; // eax
  __int64 v78; // [rsp+8h] [rbp-D8h]
  __int64 v79; // [rsp+8h] [rbp-D8h]
  unsigned int v80; // [rsp+8h] [rbp-D8h]
  __int64 v81; // [rsp+8h] [rbp-D8h]
  unsigned int v82; // [rsp+8h] [rbp-D8h]
  __int64 v83; // [rsp+8h] [rbp-D8h]
  unsigned int v84; // [rsp+8h] [rbp-D8h]
  __int64 v85; // [rsp+8h] [rbp-D8h]
  __int16 v86; // [rsp+8h] [rbp-D8h]
  int v87; // [rsp+8h] [rbp-D8h]
  __int16 *v88; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v89; // [rsp+18h] [rbp-C8h]
  char *v90; // [rsp+20h] [rbp-C0h] BYREF
  __int64 i; // [rsp+28h] [rbp-B8h]
  _BYTE v92[176]; // [rsp+30h] [rbp-B0h] BYREF

  v3 = sub_16463B0(*(_QWORD *)*a1, a2);
  v4 = (_QWORD *)*a1;
  v5 = (__int64 **)v3;
  v8 = sub_1593BB0(*a1, a2, v6, v7);
  v9 = v8;
  if ( *((_BYTE *)v4 + 16) != 9 && !v8 )
    goto LABEL_3;
  if ( (_DWORD)a2 != 1 )
  {
    v24 = 1;
    while ( v4 == (_QWORD *)a1[v24] )
    {
      if ( (_DWORD)a2 == ++v24 )
        goto LABEL_23;
    }
LABEL_3:
    v10 = (__int16 **)*v4;
    v12 = sub_15958A0(*v4);
    result = 0;
    if ( !v12 )
      return result;
    v14 = *((_BYTE *)v4 + 16);
    if ( v14 == 13 )
    {
      if ( (unsigned __int8)sub_1642F90(*v4, 8) )
      {
        v15 = &a1[a2];
        v90 = v92;
        i = 0x1000000000LL;
        if ( a1 == v15 )
        {
          v22 = 0;
          v21 = v92;
LABEL_15:
          v23 = sub_16498A0(*a1);
          result = sub_1599510(v23, v21, v22);
          goto LABEL_16;
        }
        v16 = a1;
        while ( 1 )
        {
          v17 = *v16;
          if ( *(_BYTE *)(*v16 + 16) != 13 )
            goto LABEL_65;
          v18 = *(_QWORD **)(v17 + 24);
          if ( *(_DWORD *)(v17 + 32) > 0x40u )
            v18 = (_QWORD *)*v18;
          v19 = (unsigned int)i;
          if ( (unsigned int)i >= HIDWORD(i) )
          {
            sub_16CD150(&v90, v92, 0, 1);
            v19 = (unsigned int)i;
          }
          ++v16;
          v90[v19] = (char)v18;
          v20 = i + 1;
          LODWORD(i) = i + 1;
          if ( v15 == v16 )
          {
            v21 = v90;
            v22 = v20;
            goto LABEL_15;
          }
        }
      }
      if ( (unsigned __int8)sub_1642F90(*v4, 16) )
      {
        v34 = &a1[a2];
        v90 = v92;
        i = 0x1000000000LL;
        if ( a1 == v34 )
        {
          v41 = 0;
          v40 = v92;
LABEL_54:
          v42 = sub_16498A0(*a1);
          result = sub_1599550(v42, v40, v41);
          goto LABEL_16;
        }
        v35 = a1;
        while ( 1 )
        {
          v36 = *v35;
          if ( *(_BYTE *)(*v35 + 16) != 13 )
            goto LABEL_65;
          v37 = *(_QWORD **)(v36 + 24);
          if ( *(_DWORD *)(v36 + 32) > 0x40u )
            v37 = (_QWORD *)*v37;
          v38 = (unsigned int)i;
          if ( (unsigned int)i >= HIDWORD(i) )
          {
            sub_16CD150(&v90, v92, 0, 2);
            v38 = (unsigned int)i;
          }
          ++v35;
          *(_WORD *)&v90[2 * v38] = (_WORD)v37;
          v39 = i + 1;
          LODWORD(i) = i + 1;
          if ( v34 == v35 )
          {
            v40 = v90;
            v41 = v39;
            goto LABEL_54;
          }
        }
      }
      if ( (unsigned __int8)sub_1642F90(*v4, 32) )
      {
        v43 = &a1[a2];
        v44 = a1;
        v90 = v92;
        i = 0x1000000000LL;
        if ( a1 == v43 )
        {
LABEL_63:
          v48 = v90;
          v49 = (unsigned int)i;
          v50 = sub_16498A0(*a1);
          result = sub_1599580(v50, v48, v49);
          goto LABEL_16;
        }
        while ( 1 )
        {
          v45 = *v44;
          if ( *(_BYTE *)(*v44 + 16) != 13 )
            goto LABEL_65;
          v46 = *(_QWORD **)(v45 + 24);
          if ( *(_DWORD *)(v45 + 32) > 0x40u )
            v46 = (_QWORD *)*v46;
          v47 = (unsigned int)i;
          if ( (unsigned int)i >= HIDWORD(i) )
          {
            sub_16CD150(&v90, v92, 0, 4);
            v47 = (unsigned int)i;
          }
          ++v44;
          *(_DWORD *)&v90[4 * v47] = (_DWORD)v46;
          LODWORD(i) = i + 1;
          if ( v43 == v44 )
            goto LABEL_63;
        }
      }
      if ( (unsigned __int8)sub_1642F90(*v4, 64) )
      {
        v51 = &a1[a2];
        v52 = a1;
        v90 = v92;
        i = 0x1000000000LL;
        if ( a1 != v51 )
        {
          while ( 1 )
          {
            v53 = *v52;
            if ( *(_BYTE *)(*v52 + 16) != 13 )
              break;
            v54 = *(_QWORD **)(v53 + 24);
            if ( *(_DWORD *)(v53 + 32) > 0x40u )
              v54 = (_QWORD *)*v54;
            v55 = (unsigned int)i;
            if ( (unsigned int)i >= HIDWORD(i) )
            {
              sub_16CD150(&v90, v92, 0, 8);
              v55 = (unsigned int)i;
            }
            ++v52;
            *(_QWORD *)&v90[8 * v55] = v54;
            LODWORD(i) = i + 1;
            if ( v51 == v52 )
              goto LABEL_74;
          }
LABEL_65:
          result = 0;
          goto LABEL_16;
        }
LABEL_74:
        v56 = v90;
        v57 = (unsigned int)i;
        v58 = sub_16498A0(*a1);
        result = sub_15995C0(v58, v56, v57);
        goto LABEL_16;
      }
    }
    else if ( v14 == 14 )
    {
      v25 = *(_BYTE *)(*v4 + 8LL);
      switch ( v25 )
      {
        case 1:
          v59 = &a1[a2];
          v60 = a1;
          v90 = v92;
          for ( i = 0x1000000000LL; v59 != v60; ++v60 )
          {
            v81 = *v60;
            if ( *(_BYTE *)(*v60 + 16) != 14 )
              goto LABEL_65;
            v61 = sub_16982C0(v10, a2, *v60, v11);
            v10 = &v88;
            a2 = v81 + 32;
            if ( *(_QWORD *)(v81 + 32) == v61 )
              sub_169D930(&v88, a2);
            else
              sub_169D7E0(&v88, a2);
            v82 = v89;
            if ( v89 > 0x40 )
            {
              v10 = &v88;
              v76 = sub_16A57B0(&v88);
              v62 = -1;
              if ( v82 - v76 <= 0x40 )
                v62 = *v88;
            }
            else
            {
              v62 = (__int16)v88;
            }
            v63 = (unsigned int)i;
            if ( (unsigned int)i >= HIDWORD(i) )
            {
              v10 = (__int16 **)&v90;
              a2 = (__int64)v92;
              v86 = v62;
              sub_16CD150(&v90, v92, 0, 2);
              v63 = (unsigned int)i;
              v62 = v86;
            }
            *(_WORD *)&v90[2 * v63] = v62;
            LODWORD(i) = i + 1;
            if ( v89 > 0x40 )
            {
              v10 = (__int16 **)v88;
              if ( v88 )
                j_j___libc_free_0_0(v88);
            }
          }
          v64 = v90;
          v65 = (unsigned int)i;
          v66 = sub_16498A0(*a1);
          result = sub_1599600(v66, v64, v65);
          goto LABEL_16;
        case 2:
          v67 = &a1[a2];
          v68 = a1;
          v90 = v92;
          for ( i = 0x1000000000LL; v67 != v68; ++v68 )
          {
            v83 = *v68;
            if ( *(_BYTE *)(*v68 + 16) != 14 )
              goto LABEL_65;
            v69 = sub_16982C0(v10, a2, *v68, v11);
            v10 = &v88;
            a2 = v83 + 32;
            if ( *(_QWORD *)(v83 + 32) == v69 )
              sub_169D930(&v88, a2);
            else
              sub_169D7E0(&v88, a2);
            v84 = v89;
            if ( v89 > 0x40 )
            {
              v10 = &v88;
              v77 = sub_16A57B0(&v88);
              v70 = -1;
              if ( v84 - v77 <= 0x40 )
                v70 = *(_DWORD *)v88;
            }
            else
            {
              v70 = (int)v88;
            }
            v71 = (unsigned int)i;
            if ( (unsigned int)i >= HIDWORD(i) )
            {
              v10 = (__int16 **)&v90;
              a2 = (__int64)v92;
              v87 = v70;
              sub_16CD150(&v90, v92, 0, 4);
              v71 = (unsigned int)i;
              v70 = v87;
            }
            *(_DWORD *)&v90[4 * v71] = v70;
            LODWORD(i) = i + 1;
            if ( v89 > 0x40 )
            {
              v10 = (__int16 **)v88;
              if ( v88 )
                j_j___libc_free_0_0(v88);
            }
          }
          v72 = v90;
          v73 = (unsigned int)i;
          v74 = sub_16498A0(*a1);
          result = sub_1599630(v74, v72, v73);
LABEL_16:
          if ( v90 != v92 )
          {
            v78 = result;
            _libc_free((unsigned __int64)v90);
            return v78;
          }
          return result;
        case 3:
          v26 = &a1[a2];
          v27 = a1;
          v90 = v92;
          i = 0x1000000000LL;
          if ( a1 == v26 )
          {
LABEL_42:
            v31 = v90;
            v32 = (unsigned int)i;
            v33 = sub_16498A0(*a1);
            result = sub_1599670(v33, v31, v32);
            goto LABEL_16;
          }
          while ( 1 )
          {
            v79 = *v27;
            if ( *(_BYTE *)(*v27 + 16) != 14 )
              goto LABEL_65;
            v28 = sub_16982C0(v10, a2, *v27, v11);
            v10 = &v88;
            a2 = v79 + 32;
            if ( *(_QWORD *)(v79 + 32) == v28 )
              sub_169D930(&v88, a2);
            else
              sub_169D7E0(&v88, a2);
            v80 = v89;
            if ( v89 > 0x40 )
            {
              v10 = &v88;
              v75 = sub_16A57B0(&v88);
              v29 = -1;
              if ( v80 - v75 <= 0x40 )
                v29 = *(_QWORD *)v88;
            }
            else
            {
              v29 = (__int64)v88;
            }
            v30 = (unsigned int)i;
            if ( (unsigned int)i >= HIDWORD(i) )
            {
              v10 = (__int16 **)&v90;
              a2 = (__int64)v92;
              v85 = v29;
              sub_16CD150(&v90, v92, 0, 8);
              v30 = (unsigned int)i;
              v29 = v85;
            }
            *(_QWORD *)&v90[8 * v30] = v29;
            LODWORD(i) = i + 1;
            if ( v89 > 0x40 )
            {
              v10 = (__int16 **)v88;
              if ( v88 )
                j_j___libc_free_0_0(v88);
            }
            if ( v26 == ++v27 )
              goto LABEL_42;
          }
      }
    }
    return 0;
  }
LABEL_23:
  if ( v9 )
    return sub_1598F00(v5);
  else
    return sub_1599EF0(v5);
}
