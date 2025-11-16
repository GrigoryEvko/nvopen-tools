// Function: sub_BD0200
// Address: 0xbd0200
//
_QWORD *__fastcall sub_BD0200(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 *v4; // r12
  __int64 v5; // r15
  __int64 *v6; // rbx
  int i; // eax
  __int64 v8; // rcx
  _QWORD *v9; // r11
  int v10; // r13d
  unsigned int v11; // edx
  _QWORD *v12; // rdi
  __int64 v13; // r8
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // r13
  unsigned __int64 v17; // rdx
  _BYTE *v18; // rdi
  int v19; // r13d
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v22; // rbx
  __int64 *v23; // r12
  __int64 *v24; // r11
  int v25; // r14d
  unsigned int v26; // edx
  __int64 *v27; // r8
  __int64 v28; // rdi
  int v29; // eax
  __int64 v30; // r14
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v34; // rdx
  __int64 v35; // rdi
  int v36; // r10d
  __int64 *v37; // r9
  int v38; // r10d
  __int64 v39; // rdx
  __int64 v40; // rdi
  __int128 v41; // rax
  unsigned int v42; // ebx
  __int64 v43; // rdx
  __int64 v44; // r12
  __int64 v45; // rdi
  __int64 v46; // rdx
  int v47; // r8d
  int v48; // r8d
  __int64 v49; // rdx
  _QWORD v51[2]; // [rsp+30h] [rbp-130h] BYREF
  __int64 v52; // [rsp+40h] [rbp-120h] BYREF
  _OWORD v53[2]; // [rsp+50h] [rbp-110h] BYREF
  __int64 v54; // [rsp+70h] [rbp-F0h]
  const char *v55; // [rsp+80h] [rbp-E0h]
  __int64 v56; // [rsp+88h] [rbp-D8h]
  char v57; // [rsp+A0h] [rbp-C0h]
  char v58; // [rsp+A1h] [rbp-BFh]
  __int128 v59; // [rsp+B0h] [rbp-B0h] BYREF
  const char *v60; // [rsp+C0h] [rbp-A0h]
  __int64 v61; // [rsp+C8h] [rbp-98h]
  __int64 v62; // [rsp+D0h] [rbp-90h]
  __int64 v63; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v64; // [rsp+E8h] [rbp-78h]
  __int64 v65; // [rsp+F0h] [rbp-70h]
  __int64 v66; // [rsp+F8h] [rbp-68h]
  _QWORD *v67; // [rsp+100h] [rbp-60h] BYREF
  __int64 v68; // [rsp+108h] [rbp-58h]
  _BYTE v69[80]; // [rsp+110h] [rbp-50h] BYREF

  v4 = &a3[a4];
  v67 = v69;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v68 = 0x400000000LL;
  if ( v4 == a3 )
  {
    v18 = v69;
    goto LABEL_43;
  }
  v5 = a2;
  v6 = a3;
  for ( i = 0; ; i = v65 )
  {
    if ( !i )
    {
      a2 = (__int64)&v67[(unsigned int)v68];
      if ( (_QWORD *)a2 == sub_BCAB80(v67, a2, v6) )
      {
        a2 = *v6;
        sub_BCFF80((__int64)&v63, *v6);
      }
      goto LABEL_5;
    }
    a2 = (unsigned int)v66;
    if ( !(_DWORD)v66 )
    {
      ++v63;
      goto LABEL_72;
    }
    v8 = *v6;
    v9 = 0;
    v10 = 1;
    v11 = (v66 - 1) & (((unsigned int)*v6 >> 9) ^ ((unsigned int)*v6 >> 4));
    v12 = (_QWORD *)(v64 + 8LL * v11);
    v13 = *v12;
    if ( *v6 != *v12 )
      break;
LABEL_5:
    if ( v4 == ++v6 )
      goto LABEL_20;
LABEL_6:
    ;
  }
  while ( v13 != -4096 )
  {
    if ( v9 || v13 != -8192 )
      v12 = v9;
    v11 = (v66 - 1) & (v10 + v11);
    v13 = *(_QWORD *)(v64 + 8LL * v11);
    if ( v8 == v13 )
      goto LABEL_5;
    ++v10;
    v9 = v12;
    v12 = (_QWORD *)(v64 + 8LL * v11);
  }
  if ( !v9 )
    v9 = v12;
  v14 = i + 1;
  ++v63;
  if ( 4 * v14 < (unsigned int)(3 * v66) )
  {
    if ( (int)v66 - HIDWORD(v65) - v14 <= (unsigned int)v66 >> 3 )
    {
      sub_BCFDB0((__int64)&v63, v66);
      if ( !(_DWORD)v66 )
      {
LABEL_103:
        LODWORD(v65) = v65 + 1;
        BUG();
      }
      v45 = *v6;
      a2 = 0;
      v48 = 1;
      LODWORD(v49) = (v66 - 1) & (((unsigned int)*v6 >> 9) ^ ((unsigned int)*v6 >> 4));
      v9 = (_QWORD *)(v64 + 8LL * (unsigned int)v49);
      v8 = *v9;
      v14 = v65 + 1;
      if ( *v6 != *v9 )
      {
        while ( v8 != -4096 )
        {
          if ( !a2 && v8 == -8192 )
            a2 = (__int64)v9;
          v49 = ((_DWORD)v66 - 1) & (unsigned int)(v49 + v48);
          v9 = (_QWORD *)(v64 + 8 * v49);
          v8 = *v9;
          if ( v45 == *v9 )
            goto LABEL_15;
          ++v48;
        }
LABEL_84:
        v8 = v45;
        if ( a2 )
          v9 = (_QWORD *)a2;
        goto LABEL_15;
      }
    }
    goto LABEL_15;
  }
LABEL_72:
  a2 = (unsigned int)(2 * v66);
  sub_BCFDB0((__int64)&v63, a2);
  if ( !(_DWORD)v66 )
    goto LABEL_103;
  v45 = *v6;
  LODWORD(v46) = (v66 - 1) & (((unsigned int)*v6 >> 9) ^ ((unsigned int)*v6 >> 4));
  v9 = (_QWORD *)(v64 + 8LL * (unsigned int)v46);
  v8 = *v9;
  v14 = v65 + 1;
  if ( *v6 != *v9 )
  {
    v47 = 1;
    a2 = 0;
    while ( v8 != -4096 )
    {
      if ( !a2 && v8 == -8192 )
        a2 = (__int64)v9;
      v46 = ((_DWORD)v66 - 1) & (unsigned int)(v46 + v47);
      v9 = (_QWORD *)(v64 + 8 * v46);
      v8 = *v9;
      if ( v45 == *v9 )
        goto LABEL_15;
      ++v47;
    }
    goto LABEL_84;
  }
LABEL_15:
  LODWORD(v65) = v14;
  if ( *v9 != -4096 )
    --HIDWORD(v65);
  *v9 = v8;
  v15 = (unsigned int)v68;
  v16 = *v6;
  v17 = (unsigned int)v68 + 1LL;
  if ( v17 > HIDWORD(v68) )
  {
    a2 = (__int64)v69;
    sub_C8D5F0(&v67, v69, v17, 8);
    v15 = (unsigned int)v68;
  }
  ++v6;
  v67[v15] = v16;
  LODWORD(v68) = v68 + 1;
  if ( v4 != v6 )
    goto LABEL_6;
LABEL_20:
  v18 = v67;
  if ( !(_DWORD)v68 )
  {
LABEL_43:
    *a1 = 1;
    goto LABEL_44;
  }
  v19 = 0;
  v20 = 0;
  while ( 2 )
  {
    v21 = *(_QWORD *)&v18[8 * v20];
    if ( v21 != v5 )
    {
      v22 = *(__int64 **)(v21 + 16);
      v23 = &v22[*(unsigned int *)(v21 + 12)];
      if ( v22 == v23 )
      {
LABEL_42:
        v20 = (unsigned int)(v19 + 1);
        v19 = v20;
        if ( (unsigned int)v68 <= (unsigned int)v20 )
          goto LABEL_43;
        continue;
      }
      while ( 2 )
      {
        while ( 2 )
        {
          if ( !(_DWORD)v65 )
          {
            a2 = (__int64)&v67[(unsigned int)v68];
            if ( (_QWORD *)a2 == sub_BCAB80(v67, a2, v22) )
            {
              a2 = *v22;
              sub_BCFF80((__int64)&v63, *v22);
            }
LABEL_27:
            if ( ++v22 == v23 )
              goto LABEL_41;
            continue;
          }
          break;
        }
        a2 = (unsigned int)v66;
        if ( (_DWORD)v66 )
        {
          v24 = 0;
          v25 = 1;
          v26 = (v66 - 1) & (((unsigned int)*v22 >> 9) ^ ((unsigned int)*v22 >> 4));
          v27 = (__int64 *)(v64 + 8LL * v26);
          v28 = *v27;
          if ( *v27 == *v22 )
            goto LABEL_27;
          while ( v28 != -4096 )
          {
            if ( v28 != -8192 || v24 )
              v27 = v24;
            v26 = (v66 - 1) & (v25 + v26);
            v28 = *(_QWORD *)(v64 + 8LL * v26);
            if ( *v22 == v28 )
              goto LABEL_27;
            ++v25;
            v24 = v27;
            v27 = (__int64 *)(v64 + 8LL * v26);
          }
          if ( !v24 )
            v24 = v27;
          v29 = v65 + 1;
          ++v63;
          if ( 4 * ((int)v65 + 1) < (unsigned int)(3 * v66) )
          {
            if ( (int)v66 - HIDWORD(v65) - v29 <= (unsigned int)v66 >> 3 )
            {
              sub_BCFDB0((__int64)&v63, v66);
              if ( !(_DWORD)v66 )
              {
LABEL_104:
                LODWORD(v65) = v65 + 1;
                BUG();
              }
              a2 = *v22;
              v38 = 1;
              v37 = 0;
              LODWORD(v39) = (v66 - 1) & (((unsigned int)*v22 >> 9) ^ ((unsigned int)*v22 >> 4));
              v24 = (__int64 *)(v64 + 8LL * (unsigned int)v39);
              v40 = *v24;
              v29 = v65 + 1;
              if ( *v22 != *v24 )
              {
                while ( v40 != -4096 )
                {
                  if ( v40 == -8192 && !v37 )
                    v37 = v24;
                  v39 = ((_DWORD)v66 - 1) & (unsigned int)(v39 + v38);
                  v24 = (__int64 *)(v64 + 8 * v39);
                  v40 = *v24;
                  if ( a2 == *v24 )
                    goto LABEL_36;
                  ++v38;
                }
                goto LABEL_52;
              }
            }
            goto LABEL_36;
          }
        }
        else
        {
          ++v63;
        }
        sub_BCFDB0((__int64)&v63, 2 * v66);
        if ( !(_DWORD)v66 )
          goto LABEL_104;
        a2 = *v22;
        LODWORD(v34) = (v66 - 1) & (((unsigned int)*v22 >> 9) ^ ((unsigned int)*v22 >> 4));
        v24 = (__int64 *)(v64 + 8LL * (unsigned int)v34);
        v35 = *v24;
        v29 = v65 + 1;
        if ( *v24 != *v22 )
        {
          v36 = 1;
          v37 = 0;
          while ( v35 != -4096 )
          {
            if ( !v37 && v35 == -8192 )
              v37 = v24;
            v34 = ((_DWORD)v66 - 1) & (unsigned int)(v34 + v36);
            v24 = (__int64 *)(v64 + 8 * v34);
            v35 = *v24;
            if ( a2 == *v24 )
              goto LABEL_36;
            ++v36;
          }
LABEL_52:
          if ( v37 )
            v24 = v37;
        }
LABEL_36:
        LODWORD(v65) = v29;
        if ( *v24 != -4096 )
          --HIDWORD(v65);
        v30 = *v22;
        *v24 = *v22;
        v31 = (unsigned int)v68;
        v32 = (unsigned int)v68 + 1LL;
        if ( v32 > HIDWORD(v68) )
        {
          a2 = (__int64)v69;
          sub_C8D5F0(&v67, v69, v32, 8);
          v31 = (unsigned int)v68;
        }
        ++v22;
        v67[v31] = v30;
        LODWORD(v68) = v68 + 1;
        if ( v22 == v23 )
        {
LABEL_41:
          v18 = v67;
          goto LABEL_42;
        }
        continue;
      }
    }
    break;
  }
  v58 = 1;
  v55 = "' is recursive";
  v57 = 3;
  *(_QWORD *)&v41 = sub_BCB490(v5);
  v53[1] = v41;
  *(_QWORD *)&v53[0] = "identified structure type '";
  LOWORD(v54) = 1283;
  v60 = "' is recursive";
  *(_QWORD *)&v59 = v53;
  LOWORD(v62) = 770;
  v61 = v56;
  v42 = sub_C63BB0(v5, a2, "' is recursive", v56);
  v44 = v43;
  sub_CA0F50(v51, &v59);
  a2 = (__int64)v51;
  sub_C63F00(a1, v51, v42, v44);
  if ( (__int64 *)v51[0] != &v52 )
  {
    a2 = v52 + 1;
    j_j___libc_free_0(v51[0], v52 + 1);
  }
  v18 = v67;
LABEL_44:
  if ( v18 != v69 )
    _libc_free(v18, a2);
  sub_C7D6A0(v64, 8LL * (unsigned int)v66, 8);
  return a1;
}
