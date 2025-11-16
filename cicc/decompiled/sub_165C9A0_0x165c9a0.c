// Function: sub_165C9A0
// Address: 0x165c9a0
//
__int64 __fastcall sub_165C9A0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r15
  unsigned int v5; // r14d
  int v6; // r12d
  __int64 v7; // r8
  _BYTE *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r13
  unsigned __int8 **v12; // rdx
  __int64 v13; // r13
  unsigned __int8 *v14; // rax
  _BYTE *v15; // rax
  unsigned __int8 **v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rdi
  _BYTE *v19; // rax
  _BYTE *v20; // r15
  _BYTE *v21; // r13
  __int64 v22; // r14
  unsigned int v23; // ecx
  unsigned __int8 **v24; // rax
  unsigned __int8 *v25; // r10
  unsigned __int8 *v26; // rdx
  unsigned __int8 *v27; // rbx
  const char *v28; // rax
  __int64 v29; // r8
  _BYTE *v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdi
  _BYTE *v33; // rax
  __int64 v34; // rdx
  char *v35; // r8
  unsigned __int8 **v36; // rdx
  unsigned __int8 *v37; // r9
  const char *v38; // rax
  unsigned __int8 *v39; // rax
  int v40; // esi
  int v41; // r11d
  const char **v42; // r10
  unsigned int v43; // edx
  const char **v44; // rax
  char *v45; // rcx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // r8
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rsi
  unsigned __int8 **v56; // r13
  unsigned __int8 **v57; // r12
  _BYTE *v58; // rax
  __int64 v59; // rsi
  __int64 v60; // rdi
  _BYTE *v61; // rax
  unsigned __int8 *v62; // r15
  __int64 v63; // rdx
  __int64 v64; // r15
  int v65; // eax
  int v66; // r8d
  __int64 v67; // rax
  __int64 v68; // rdi
  __int64 v69; // rax
  const char *v70; // rcx
  int v71; // edx
  unsigned __int8 *v72; // rcx
  unsigned __int8 *v73; // rcx
  unsigned __int8 **v74; // rdx
  unsigned __int8 *v75; // rax
  unsigned __int8 *v76; // rsi
  unsigned __int8 **v77; // [rsp+8h] [rbp-138h]
  int v78; // [rsp+20h] [rbp-120h]
  char *v79; // [rsp+20h] [rbp-120h]
  unsigned __int8 *v80; // [rsp+20h] [rbp-120h]
  __int64 v81; // [rsp+28h] [rbp-118h]
  unsigned __int8 **v82; // [rsp+28h] [rbp-118h]
  __int64 v83; // [rsp+28h] [rbp-118h]
  unsigned __int8 *v84; // [rsp+28h] [rbp-118h]
  __int64 v85; // [rsp+28h] [rbp-118h]
  __int64 v86; // [rsp+28h] [rbp-118h]
  char *v87; // [rsp+28h] [rbp-118h]
  unsigned int v88; // [rsp+34h] [rbp-10Ch] BYREF
  const char **v89; // [rsp+38h] [rbp-108h] BYREF
  const char *v90; // [rsp+40h] [rbp-100h] BYREF
  const char *v91; // [rsp+48h] [rbp-F8h]
  char v92; // [rsp+50h] [rbp-F0h]
  char v93; // [rsp+51h] [rbp-EFh]
  __int64 v94; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v95; // [rsp+68h] [rbp-D8h]
  __int64 v96; // [rsp+70h] [rbp-D0h]
  unsigned int v97; // [rsp+78h] [rbp-C8h]
  _BYTE *v98; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v99; // [rsp+88h] [rbp-B8h]
  _BYTE v100[176]; // [rsp+90h] [rbp-B0h] BYREF

  result = sub_16327A0(a2);
  if ( !result )
    return result;
  v4 = result;
  v5 = 0;
  v94 = 0;
  v98 = v100;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v99 = 0x1000000000LL;
  v6 = sub_161F520(result);
  if ( v6 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = sub_161F530(v4, v5);
        v11 = v10;
        if ( *(_DWORD *)(v10 + 8) != 3 )
        {
          v7 = *(_QWORD *)a1;
          v93 = 1;
          v90 = "incorrect number of operands in module flag";
          v92 = 3;
          if ( !v7 )
            goto LABEL_44;
          v81 = v7;
          sub_16E2CE0(&v90, v7);
          v8 = *(_BYTE **)(v81 + 24);
          if ( (unsigned __int64)v8 >= *(_QWORD *)(v81 + 16) )
          {
            sub_16E7DE0(v81, 10);
          }
          else
          {
            *(_QWORD *)(v81 + 24) = v8 + 1;
            *v8 = 10;
          }
          v9 = *(_QWORD *)a1;
          *(_BYTE *)(a1 + 72) = 1;
          if ( v9 )
            sub_164ED40((__int64 *)a1, (unsigned __int8 *)v11);
          goto LABEL_9;
        }
        if ( (unsigned __int8)sub_1632720(*(_QWORD *)(v10 - 24), &v88) )
          break;
        v12 = (unsigned __int8 **)(v11 - 8LL * *(unsigned int *)(v11 + 8));
        v13 = *(_QWORD *)a1;
        v14 = *v12;
        if ( *v12 && *v14 == 1 && *(_BYTE *)(*((_QWORD *)v14 + 17) + 16LL) == 13 )
        {
          v93 = 1;
          v90 = "invalid behavior operand in module flag (unexpected constant)";
          v92 = 3;
          if ( !v13 )
            goto LABEL_44;
          v82 = v12;
        }
        else
        {
          v82 = v12;
          v93 = 1;
          v90 = "invalid behavior operand in module flag (expected constant integer)";
          v92 = 3;
          if ( !v13 )
          {
LABEL_44:
            *(_BYTE *)(a1 + 72) = 1;
            goto LABEL_9;
          }
        }
        sub_16E2CE0(&v90, v13);
        v15 = *(_BYTE **)(v13 + 24);
        v16 = v82;
        if ( (unsigned __int64)v15 >= *(_QWORD *)(v13 + 16) )
        {
          sub_16E7DE0(v13, 10);
          v17 = *(_QWORD *)a1;
          v16 = v82;
        }
        else
        {
          *(_QWORD *)(v13 + 24) = v15 + 1;
          *v15 = 10;
          v17 = *(_QWORD *)a1;
        }
        *(_BYTE *)(a1 + 72) = 1;
        if ( !v17 || !*v16 )
          goto LABEL_9;
        sub_15562E0(*v16, v17, a1 + 16, *(_QWORD *)(a1 + 8));
        v18 = *(_QWORD *)a1;
        v19 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
        if ( (unsigned __int64)v19 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        {
          sub_16E7DE0(v18, 10);
          goto LABEL_9;
        }
        ++v5;
        *(_QWORD *)(v18 + 24) = v19 + 1;
        *v19 = 10;
        if ( v6 == v5 )
          goto LABEL_23;
      }
      v34 = *(unsigned int *)(v11 + 8);
      v35 = *(char **)(v11 + 8 * (1 - v34));
      if ( !v35 || *v35 )
      {
        v93 = 1;
        v90 = "invalid ID operand in module flag (expected metadata string)";
        v92 = 3;
        sub_16589C0((_BYTE *)a1, (__int64)&v90, (unsigned __int8 **)(v11 + 8 * (1 - v34)));
        goto LABEL_9;
      }
      if ( v88 > 6 )
      {
        if ( v88 == 7 )
        {
          v36 = (unsigned __int8 **)(v11 + 8 * (2 - v34));
          v39 = *v36;
          if ( !*v36 || *v39 != 1 || *(_BYTE *)(*((_QWORD *)v39 + 17) + 16LL) != 13 )
          {
            v93 = 1;
            v38 = "invalid value for 'max' module flag (expected constant integer)";
            goto LABEL_52;
          }
        }
      }
      else if ( v88 > 4 )
      {
        v36 = (unsigned __int8 **)(v11 + 8 * (2 - v34));
        if ( (unsigned __int8)(**v36 - 4) > 0x1Eu )
        {
          v93 = 1;
          v38 = "invalid value for 'append'-type module flag (expected a metadata node)";
          goto LABEL_52;
        }
      }
      else if ( v88 == 3 )
      {
        v36 = (unsigned __int8 **)(v11 + 8 * (2 - v34));
        v37 = *v36;
        if ( (unsigned __int8)(**v36 - 4) > 0x1Eu || *((_DWORD *)v37 + 2) != 2 )
        {
          v93 = 1;
          v38 = "invalid value for 'require' module flag (expected metadata pair)";
LABEL_52:
          v90 = v38;
          v92 = 3;
          sub_16589C0((_BYTE *)a1, (__int64)&v90, v36);
          goto LABEL_9;
        }
        if ( **((_BYTE **)v37 - 2) )
        {
          v93 = 1;
          v90 = "invalid value for 'require' module flag (first value operand should be a string)";
          v92 = 3;
          sub_16589C0((_BYTE *)a1, (__int64)&v90, (unsigned __int8 **)v37 - 2);
          goto LABEL_9;
        }
        v46 = (unsigned int)v99;
        if ( (unsigned int)v99 >= HIDWORD(v99) )
        {
          v80 = *v36;
          v87 = v35;
          sub_16CD150(&v98, v100, 0, 8);
          v46 = (unsigned int)v99;
          v37 = v80;
          v35 = v87;
        }
        *(_QWORD *)&v98[8 * v46] = v37;
        LODWORD(v99) = v99 + 1;
        if ( v88 == 3 )
          goto LABEL_71;
      }
      v40 = v97;
      v90 = v35;
      v91 = (const char *)v11;
      if ( !v97 )
        break;
      v41 = 1;
      v42 = 0;
      v43 = (v97 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v44 = (const char **)(v95 + 16LL * v43);
      v45 = (char *)*v44;
      if ( v35 == *v44 )
      {
LABEL_63:
        v84 = (unsigned __int8 *)v35;
        v93 = 1;
        v90 = "module flag identifiers must be unique (or of 'require' type)";
        v92 = 3;
        sub_164FF40((__int64 *)a1, (__int64)&v90);
        if ( *(_QWORD *)a1 )
          sub_164ED40((__int64 *)a1, v84);
        goto LABEL_9;
      }
      while ( v45 != (char *)-8LL )
      {
        if ( v42 || v45 != (char *)-16LL )
          v44 = v42;
        v43 = (v97 - 1) & (v41 + v43);
        v45 = *(char **)(v95 + 16LL * v43);
        if ( v35 == v45 )
          goto LABEL_63;
        ++v41;
        v42 = v44;
        v44 = (const char **)(v95 + 16LL * v43);
      }
      if ( !v42 )
        v42 = v44;
      ++v94;
      v71 = v96 + 1;
      if ( 4 * ((int)v96 + 1) >= 3 * v97 )
        goto LABEL_113;
      v70 = v35;
      if ( v97 - HIDWORD(v96) - v71 > v97 >> 3 )
        goto LABEL_115;
      v79 = v35;
LABEL_114:
      sub_165C7E0((__int64)&v94, v40);
      sub_165C1D0((__int64)&v94, (__int64 *)&v90, &v89);
      v42 = v89;
      v70 = v90;
      v35 = v79;
      v71 = v96 + 1;
LABEL_115:
      LODWORD(v96) = v71;
      if ( *v42 != (const char *)-8LL )
        --HIDWORD(v96);
      *v42 = v70;
      v42[1] = v91;
LABEL_71:
      v85 = (__int64)v35;
      v47 = sub_161E970((__int64)v35);
      if ( v48 != 10
        || *(_QWORD *)v47 != 0x69735F7261686377LL
        || *(_WORD *)(v47 + 8) != 25978
        || (v67 = *(_QWORD *)(v11 + 8 * (2LL - *(unsigned int *)(v11 + 8)))) != 0
        && *(_BYTE *)v67 == 1
        && *(_BYTE *)(*(_QWORD *)(v67 + 136) + 16LL) == 13 )
      {
        v49 = sub_161E970(v85);
        v50 = v85;
        if ( v51 != 14
          || *(_QWORD *)v49 != 0x4F2072656B6E694CLL
          || *(_DWORD *)(v49 + 8) != 1869182064
          || *(_WORD *)(v49 + 12) != 29550
          || (v68 = *(_QWORD *)(a1 + 8),
              v93 = 1,
              v90 = "llvm.linker.options",
              v92 = 3,
              v69 = sub_1632310(v68, (__int64)&v90),
              v50 = v85,
              v69) )
        {
          v52 = sub_161E970(v50);
          if ( v53 != 10 )
            goto LABEL_9;
          if ( *(_QWORD *)v52 != 0x69666F7250204743LL )
            goto LABEL_9;
          if ( *(_WORD *)(v52 + 8) != 25964 )
            goto LABEL_9;
          v54 = *(unsigned int *)(v11 + 8);
          v55 = *(_QWORD *)(v11 + 8 * (2 - v54));
          if ( v55 == v55 - 8LL * *(unsigned int *)(v55 + 8) )
            goto LABEL_9;
          v78 = v6;
          v56 = *(unsigned __int8 ***)(v11 + 8 * (2 - v54));
          v57 = (unsigned __int8 **)(v55 - 8LL * *(unsigned int *)(v55 + 8));
          v86 = v4;
          while ( 2 )
          {
            while ( 1 )
            {
              v62 = *v57;
              if ( *v57 )
              {
                if ( (unsigned __int8)(*v62 - 4) <= 0x1Eu )
                {
                  v63 = *((unsigned int *)v62 + 2);
                  if ( (_DWORD)v63 == 3 )
                    break;
                }
              }
              v64 = *(_QWORD *)a1;
              v93 = 1;
              v90 = "expected a MDNode triple";
              v92 = 3;
              if ( v64 )
              {
                sub_16E2CE0(&v90, v64);
                v58 = *(_BYTE **)(v64 + 24);
                if ( (unsigned __int64)v58 >= *(_QWORD *)(v64 + 16) )
                {
                  sub_16E7DE0(v64, 10);
                }
                else
                {
                  *(_QWORD *)(v64 + 24) = v58 + 1;
                  *v58 = 10;
                }
                v59 = *(_QWORD *)a1;
                *(_BYTE *)(a1 + 72) = 1;
                if ( v59 && *v57 )
                {
                  sub_15562E0(*v57, v59, a1 + 16, *(_QWORD *)(a1 + 8));
                  v60 = *(_QWORD *)a1;
                  v61 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
                  if ( (unsigned __int64)v61 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
                  {
                    sub_16E7DE0(v60, 10);
                  }
                  else
                  {
                    *(_QWORD *)(v60 + 24) = v61 + 1;
                    *v61 = 10;
                  }
                }
                goto LABEL_84;
              }
              ++v57;
              *(_BYTE *)(a1 + 72) = 1;
              if ( v56 == v57 )
              {
LABEL_90:
                v4 = v86;
                v6 = v78;
                goto LABEL_9;
              }
            }
            v72 = (unsigned __int8 *)*((_QWORD *)v62 - 3);
            if ( v72 && ((unsigned int)*v72 - 1 > 1 || *(_BYTE *)(*((_QWORD *)v72 + 17) + 16LL)) )
            {
              v93 = 1;
              v90 = "expected a Function or null";
              v92 = 3;
              sub_164FF40((__int64 *)a1, (__int64)&v90);
              if ( *(_QWORD *)a1 )
              {
                v76 = (unsigned __int8 *)*((_QWORD *)v62 - 3);
                if ( v76 )
                  sub_164ED40((__int64 *)a1, v76);
              }
              v63 = *((unsigned int *)v62 + 2);
            }
            v73 = *(unsigned __int8 **)&v62[8 * (1 - v63)];
            if ( v73 && ((unsigned int)*v73 - 1 > 1 || *(_BYTE *)(*((_QWORD *)v73 + 17) + 16LL)) )
            {
              v77 = (unsigned __int8 **)&v62[8 * (1 - v63)];
              v93 = 1;
              v90 = "expected a Function or null";
              v92 = 3;
              sub_164FF40((__int64 *)a1, (__int64)&v90);
              if ( *(_QWORD *)a1 && *v77 )
                sub_164ED40((__int64 *)a1, *v77);
              v63 = *((unsigned int *)v62 + 2);
            }
            v74 = (unsigned __int8 **)&v62[8 * (2 - v63)];
            v75 = *v74;
            if ( !*v74 || *v75 != 1 || *(_BYTE *)(**((_QWORD **)v75 + 17) + 8LL) != 11 )
            {
              v93 = 1;
              v90 = "expected an integer constant";
              v92 = 3;
              sub_16589C0((_BYTE *)a1, (__int64)&v90, v74);
            }
LABEL_84:
            if ( v56 == ++v57 )
              goto LABEL_90;
            continue;
          }
        }
        v93 = 1;
        v90 = "'Linker Options' named metadata no longer supported";
        v92 = 3;
        sub_164FF40((__int64 *)a1, (__int64)&v90);
      }
      else
      {
        v93 = 1;
        v90 = "wchar_size metadata requires constant integer argument";
        v92 = 3;
        sub_164FF40((__int64 *)a1, (__int64)&v90);
      }
LABEL_9:
      if ( v6 == ++v5 )
        goto LABEL_23;
    }
    ++v94;
LABEL_113:
    v79 = v35;
    v40 = 2 * v97;
    goto LABEL_114;
  }
LABEL_23:
  v20 = &v98[8 * (unsigned int)v99];
  if ( v98 == v20 )
    goto LABEL_39;
  v21 = v98;
  v22 = a1;
  do
  {
    while ( 1 )
    {
      v27 = *(unsigned __int8 **)(*(_QWORD *)v21 - 8LL * *(unsigned int *)(*(_QWORD *)v21 + 8LL));
      if ( !v97 )
        goto LABEL_30;
      v23 = (v97 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v24 = (unsigned __int8 **)(v95 + 16LL * v23);
      v25 = *v24;
      if ( v27 != *v24 )
      {
        v65 = 1;
        while ( v25 != (unsigned __int8 *)-8LL )
        {
          v66 = v65 + 1;
          v23 = (v97 - 1) & (v65 + v23);
          v24 = (unsigned __int8 **)(v95 + 16LL * v23);
          v25 = *v24;
          if ( v27 == *v24 )
            goto LABEL_26;
          v65 = v66;
        }
LABEL_30:
        v93 = 1;
        v28 = "invalid requirement on flag, flag is not present in module";
        goto LABEL_31;
      }
LABEL_26:
      v26 = v24[1];
      if ( !v26 )
        goto LABEL_30;
      if ( *(_QWORD *)(*(_QWORD *)v21 + 8 * (1LL - *(unsigned int *)(*(_QWORD *)v21 + 8LL))) != *(_QWORD *)&v26[8 * (2LL - *((unsigned int *)v26 + 2))] )
        break;
LABEL_28:
      v21 += 8;
      if ( v20 == v21 )
        goto LABEL_38;
    }
    v93 = 1;
    v28 = "invalid requirement on flag, flag does not have the required value";
LABEL_31:
    v29 = *(_QWORD *)v22;
    v90 = v28;
    v92 = 3;
    if ( !v29 )
    {
      *(_BYTE *)(v22 + 72) = 1;
      goto LABEL_28;
    }
    v83 = v29;
    sub_16E2CE0(&v90, v29);
    v30 = *(_BYTE **)(v83 + 24);
    if ( (unsigned __int64)v30 >= *(_QWORD *)(v83 + 16) )
    {
      sub_16E7DE0(v83, 10);
    }
    else
    {
      *(_QWORD *)(v83 + 24) = v30 + 1;
      *v30 = 10;
    }
    v31 = *(_QWORD *)v22;
    *(_BYTE *)(v22 + 72) = 1;
    if ( !v31 || !v27 )
      goto LABEL_28;
    sub_15562E0(v27, v31, v22 + 16, *(_QWORD *)(v22 + 8));
    v32 = *(_QWORD *)v22;
    v33 = *(_BYTE **)(*(_QWORD *)v22 + 24LL);
    if ( (unsigned __int64)v33 >= *(_QWORD *)(*(_QWORD *)v22 + 16LL) )
    {
      sub_16E7DE0(v32, 10);
      goto LABEL_28;
    }
    v21 += 8;
    *(_QWORD *)(v32 + 24) = v33 + 1;
    *v33 = 10;
  }
  while ( v20 != v21 );
LABEL_38:
  v20 = v98;
LABEL_39:
  if ( v20 != v100 )
    _libc_free((unsigned __int64)v20);
  return j___libc_free_0(v95);
}
