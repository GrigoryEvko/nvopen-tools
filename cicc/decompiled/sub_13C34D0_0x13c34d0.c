// Function: sub_13C34D0
// Address: 0x13c34d0
//
__int64 __fastcall sub_13C34D0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r14
  unsigned __int64 v5; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  int v16; // r12d
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r12
  int v28; // r12d
  __int64 v29; // rax
  __int64 v30; // rdx
  _QWORD *v31; // r14
  __int64 v32; // rdx
  __int64 *v33; // r15
  __int64 v34; // r12
  __int64 *v35; // r14
  __int64 v36; // rax
  __int64 v37; // r12
  __int64 *v38; // r12
  __int64 *v39; // rdi
  __int64 v40; // rdx
  __int64 *v41; // rsi
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 *v44; // rax
  __int64 *v45; // rdx
  __int64 *v46; // r12
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // r15
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // r12
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r15
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // r12
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  unsigned __int8 v76; // [rsp+Fh] [rbp-F1h]
  __int64 *v77; // [rsp+10h] [rbp-F0h]
  _QWORD *v78; // [rsp+20h] [rbp-E0h]
  __int64 v79; // [rsp+30h] [rbp-D0h]
  _QWORD *v80; // [rsp+38h] [rbp-C8h]
  __int64 v81; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v82; // [rsp+48h] [rbp-B8h]
  __int64 v83; // [rsp+50h] [rbp-B0h]
  __int64 v84; // [rsp+58h] [rbp-A8h]
  __int64 v85; // [rsp+60h] [rbp-A0h]
  __int64 v86; // [rsp+70h] [rbp-90h] BYREF
  __int64 v87; // [rsp+78h] [rbp-88h]
  __int64 v88; // [rsp+80h] [rbp-80h]
  __int64 v89; // [rsp+88h] [rbp-78h]
  __int64 v90; // [rsp+90h] [rbp-70h]
  __int64 *v91; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v92; // [rsp+A8h] [rbp-58h]
  _BYTE v93[80]; // [rsp+B0h] [rbp-50h] BYREF

  v3 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = (a2 & 0xFFFFFFFFFFFFFFF8LL) + 56;
  if ( (a2 & 4) != 0 )
  {
    if ( (unsigned __int8)sub_1560260(v5, 0xFFFFFFFFLL, 36) )
      return 4;
    if ( *(char *)(v3 + 23) >= 0 )
      goto LABEL_158;
    v20 = sub_1648A40(v3);
    v22 = v20 + v21;
    v23 = 0;
    if ( *(char *)(v3 + 23) < 0 )
      v23 = sub_1648A40(v3);
    if ( !(unsigned int)((v22 - v23) >> 4) )
    {
LABEL_158:
      v24 = *(_QWORD *)(v3 - 24);
      if ( !*(_BYTE *)(v24 + 16) )
      {
        v91 = *(__int64 **)(v24 + 112);
        if ( (unsigned __int8)sub_1560260(&v91, 0xFFFFFFFFLL, 36) )
          return 4;
      }
    }
    if ( !(unsigned __int8)sub_1560260(v5, 0xFFFFFFFFLL, 36) )
    {
      if ( *(char *)(v3 + 23) < 0 )
      {
        v56 = sub_1648A40(v3);
        v58 = v56 + v57;
        v59 = 0;
        if ( *(char *)(v3 + 23) < 0 )
          v59 = sub_1648A40(v3);
        if ( (unsigned int)((v58 - v59) >> 4) )
          goto LABEL_159;
      }
      v60 = *(_QWORD *)(v3 - 24);
      if ( *(_BYTE *)(v60 + 16)
        || (v91 = *(__int64 **)(v60 + 112), !(unsigned __int8)sub_1560260(&v91, 0xFFFFFFFFLL, 36)) )
      {
LABEL_159:
        if ( !(unsigned __int8)sub_1560260(v5, 0xFFFFFFFFLL, 37) )
        {
          if ( *(char *)(v3 + 23) < 0 )
          {
            v61 = sub_1648A40(v3);
            v63 = v61 + v62;
            v64 = *(char *)(v3 + 23) >= 0 ? 0LL : sub_1648A40(v3);
            if ( v64 != v63 )
            {
              while ( *(_DWORD *)(*(_QWORD *)v64 + 8LL) <= 1u )
              {
                v64 += 16;
                if ( v63 == v64 )
                  goto LABEL_145;
              }
              goto LABEL_125;
            }
          }
LABEL_145:
          v75 = *(_QWORD *)(v3 - 24);
          if ( *(_BYTE *)(v75 + 16)
            || (v91 = *(__int64 **)(v75 + 112), !(unsigned __int8)sub_1560260(&v91, 0xFFFFFFFFLL, 37)) )
          {
LABEL_125:
            v76 = 7;
LABEL_28:
            if ( *(char *)(v3 + 23) < 0 )
            {
              v25 = sub_1648A40(v3);
              v27 = v25 + v26;
              if ( *(char *)(v3 + 23) >= 0 )
              {
                if ( (unsigned int)(v27 >> 4) )
                  goto LABEL_155;
              }
              else if ( (unsigned int)((v27 - sub_1648A40(v3)) >> 4) )
              {
                if ( *(char *)(v3 + 23) < 0 )
                {
                  v28 = *(_DWORD *)(sub_1648A40(v3) + 8);
                  if ( *(char *)(v3 + 23) >= 0 )
                    BUG();
                  v29 = sub_1648A40(v3);
                  v19 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v29 + v30 - 4) - v28);
                  goto LABEL_36;
                }
LABEL_155:
                BUG();
              }
            }
            v19 = -24;
            goto LABEL_36;
          }
        }
      }
    }
    v76 = 5;
    goto LABEL_28;
  }
  if ( (unsigned __int8)sub_1560260(v5, 0xFFFFFFFFLL, 36) )
    return 4;
  if ( *(char *)(v3 + 23) >= 0 )
    goto LABEL_160;
  v8 = sub_1648A40(v3);
  v10 = v8 + v9;
  v11 = 0;
  if ( *(char *)(v3 + 23) < 0 )
    v11 = sub_1648A40(v3);
  if ( !(unsigned int)((v10 - v11) >> 4) )
  {
LABEL_160:
    v12 = *(_QWORD *)(v3 - 72);
    if ( !*(_BYTE *)(v12 + 16) )
    {
      v91 = *(__int64 **)(v12 + 112);
      if ( (unsigned __int8)sub_1560260(&v91, 0xFFFFFFFFLL, 36) )
        return 4;
    }
  }
  if ( (unsigned __int8)sub_1560260(v5, 0xFFFFFFFFLL, 36) )
    goto LABEL_13;
  if ( *(char *)(v3 + 23) >= 0 )
    goto LABEL_161;
  v65 = sub_1648A40(v3);
  v67 = v65 + v66;
  v68 = 0;
  if ( *(char *)(v3 + 23) < 0 )
    v68 = sub_1648A40(v3);
  if ( !(unsigned int)((v67 - v68) >> 4) )
  {
LABEL_161:
    v69 = *(_QWORD *)(v3 - 72);
    if ( !*(_BYTE *)(v69 + 16) )
    {
      v91 = *(__int64 **)(v69 + 112);
      if ( (unsigned __int8)sub_1560260(&v91, 0xFFFFFFFFLL, 36) )
        goto LABEL_13;
    }
  }
  if ( (unsigned __int8)sub_1560260(v5, 0xFFFFFFFFLL, 37) )
    goto LABEL_13;
  if ( *(char *)(v3 + 23) < 0 )
  {
    v70 = sub_1648A40(v3);
    v72 = v70 + v71;
    v73 = *(char *)(v3 + 23) >= 0 ? 0LL : sub_1648A40(v3);
    if ( v73 != v72 )
    {
      while ( *(_DWORD *)(*(_QWORD *)v73 + 8LL) <= 1u )
      {
        v73 += 16;
        if ( v72 == v73 )
          goto LABEL_141;
      }
      goto LABEL_138;
    }
  }
LABEL_141:
  v74 = *(_QWORD *)(v3 - 72);
  if ( !*(_BYTE *)(v74 + 16) )
  {
    v91 = *(__int64 **)(v74 + 112);
    if ( (unsigned __int8)sub_1560260(&v91, 0xFFFFFFFFLL, 37) )
    {
LABEL_13:
      v76 = 5;
      goto LABEL_14;
    }
  }
LABEL_138:
  v76 = 7;
LABEL_14:
  if ( *(char *)(v3 + 23) >= 0 )
    goto LABEL_35;
  v13 = sub_1648A40(v3);
  v15 = v13 + v14;
  if ( *(char *)(v3 + 23) >= 0 )
  {
    if ( (unsigned int)(v15 >> 4) )
LABEL_153:
      BUG();
LABEL_35:
    v19 = -72;
    goto LABEL_36;
  }
  if ( !(unsigned int)((v15 - sub_1648A40(v3)) >> 4) )
    goto LABEL_35;
  if ( *(char *)(v3 + 23) >= 0 )
    goto LABEL_153;
  v16 = *(_DWORD *)(sub_1648A40(v3) + 8);
  if ( *(char *)(v3 + 23) >= 0 )
    BUG();
  v17 = sub_1648A40(v3);
  v19 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v17 + v18 - 4) - v16);
LABEL_36:
  v78 = (_QWORD *)(v3 + v19);
  v31 = (_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
  if ( v78 == v31 )
    return 4;
  v80 = v31;
  while ( 1 )
  {
    v32 = *(_QWORD *)(a1 + 8);
    v91 = (__int64 *)v93;
    v92 = 0x400000000LL;
    sub_14AD470(*v80, &v91, v32, 0, 6);
    v33 = v91;
    v34 = 8LL * (unsigned int)v92;
    v35 = &v91[(unsigned __int64)v34 / 8];
    v36 = v34 >> 3;
    v37 = v34 >> 5;
    if ( v37 )
    {
      v38 = &v91[4 * v37];
      while ( (unsigned __int8)sub_134E860(*v33) )
      {
        if ( !(unsigned __int8)sub_134E860(v33[1]) )
        {
          ++v33;
          goto LABEL_45;
        }
        if ( !(unsigned __int8)sub_134E860(v33[2]) )
        {
          v33 += 2;
          goto LABEL_45;
        }
        if ( !(unsigned __int8)sub_134E860(v33[3]) )
        {
          v33 += 3;
          goto LABEL_45;
        }
        v33 += 4;
        if ( v38 == v33 )
        {
          v36 = v35 - v33;
          goto LABEL_72;
        }
      }
      goto LABEL_45;
    }
LABEL_72:
    if ( v36 == 2 )
      goto LABEL_95;
    if ( v36 == 3 )
    {
      if ( !(unsigned __int8)sub_134E860(*v33) )
        goto LABEL_45;
      ++v33;
LABEL_95:
      if ( !(unsigned __int8)sub_134E860(*v33) )
        goto LABEL_45;
      ++v33;
      goto LABEL_75;
    }
    if ( v36 != 1 )
      goto LABEL_46;
LABEL_75:
    if ( (unsigned __int8)sub_134E860(*v33) )
      goto LABEL_46;
LABEL_45:
    if ( v35 == v33 )
      goto LABEL_46;
    v46 = v91;
    v47 = 8LL * (unsigned int)v92;
    v77 = &v91[(unsigned __int64)v47 / 8];
    v48 = v47 >> 3;
    v79 = v47 >> 5;
    if ( v47 >> 5 )
      break;
LABEL_98:
    if ( v48 != 2 )
    {
      if ( v48 != 3 )
      {
        if ( v48 != 1 )
          goto LABEL_46;
        goto LABEL_101;
      }
      v54 = *v46;
      v81 = a3;
      v82 = -1;
      v83 = 0;
      v84 = 0;
      v85 = 0;
      v86 = v54;
      v87 = -1;
      v88 = 0;
      v89 = 0;
      v90 = 0;
      if ( sub_13C3130(a1, &v86, &v81) )
        goto LABEL_67;
      ++v46;
    }
    v55 = *v46;
    v81 = a3;
    v82 = -1;
    v83 = 0;
    v84 = 0;
    v85 = 0;
    v86 = v55;
    v87 = -1;
    v88 = 0;
    v89 = 0;
    v90 = 0;
    if ( sub_13C3130(a1, &v86, &v81) )
      goto LABEL_67;
    ++v46;
LABEL_101:
    v53 = *v46;
    v81 = a3;
    v82 = -1;
    v83 = 0;
    v84 = 0;
    v85 = 0;
    v86 = v53;
    v87 = -1;
    v88 = 0;
    v89 = 0;
    v90 = 0;
    if ( sub_13C3130(a1, &v86, &v81) )
      goto LABEL_67;
LABEL_46:
    v39 = v91;
    v40 = 8LL * (unsigned int)v92;
    v41 = &v91[(unsigned __int64)v40 / 8];
    v42 = v40 >> 3;
    v43 = v40 >> 5;
    if ( v43 )
    {
      v44 = v91;
      v45 = &v91[4 * v43];
      while ( a3 != *v44 )
      {
        if ( a3 == v44[1] )
        {
          ++v44;
          goto LABEL_53;
        }
        if ( a3 == v44[2] )
        {
          v44 += 2;
          goto LABEL_53;
        }
        if ( a3 == v44[3] )
        {
          v44 += 3;
          goto LABEL_53;
        }
        v44 += 4;
        if ( v45 == v44 )
        {
          v42 = v41 - v44;
          goto LABEL_84;
        }
      }
      goto LABEL_53;
    }
    v44 = v91;
LABEL_84:
    if ( v42 == 2 )
      goto LABEL_91;
    if ( v42 != 3 )
    {
      if ( v42 != 1 )
        goto LABEL_54;
      goto LABEL_87;
    }
    if ( a3 != *v44 )
    {
      ++v44;
LABEL_91:
      if ( a3 != *v44 )
      {
        ++v44;
LABEL_87:
        if ( a3 != *v44 )
          goto LABEL_54;
      }
    }
LABEL_53:
    if ( v41 != v44 )
      goto LABEL_69;
LABEL_54:
    if ( v91 != (__int64 *)v93 )
      _libc_free((unsigned __int64)v91);
    v80 += 3;
    if ( v78 == v80 )
      return 4;
  }
  while ( 1 )
  {
    v52 = *v46;
    v81 = a3;
    v82 = -1;
    v83 = 0;
    v84 = 0;
    v85 = 0;
    v86 = v52;
    v87 = -1;
    v88 = 0;
    v89 = 0;
    v90 = 0;
    if ( sub_13C3130(a1, &v86, &v81) )
      break;
    v49 = v46[1];
    v81 = a3;
    v82 = -1;
    v83 = 0;
    v84 = 0;
    v85 = 0;
    v86 = v49;
    v87 = -1;
    v88 = 0;
    v89 = 0;
    v90 = 0;
    if ( sub_13C3130(a1, &v86, &v81) )
    {
      if ( v77 == v46 + 1 )
        goto LABEL_46;
      goto LABEL_68;
    }
    v50 = v46[2];
    v81 = a3;
    v82 = -1;
    v83 = 0;
    v84 = 0;
    v85 = 0;
    v86 = v50;
    v87 = -1;
    v88 = 0;
    v89 = 0;
    v90 = 0;
    if ( sub_13C3130(a1, &v86, &v81) )
    {
      if ( v77 == v46 + 2 )
        goto LABEL_46;
      goto LABEL_68;
    }
    v51 = v46[3];
    v81 = a3;
    v82 = -1;
    v83 = 0;
    v84 = 0;
    v85 = 0;
    v86 = v51;
    v87 = -1;
    v88 = 0;
    v89 = 0;
    v90 = 0;
    if ( sub_13C3130(a1, &v86, &v81) )
    {
      if ( v77 == v46 + 3 )
        goto LABEL_46;
      goto LABEL_68;
    }
    v46 += 4;
    if ( !--v79 )
    {
      v48 = v77 - v46;
      goto LABEL_98;
    }
  }
LABEL_67:
  if ( v77 == v46 )
    goto LABEL_46;
LABEL_68:
  v39 = v91;
LABEL_69:
  if ( v39 != (__int64 *)v93 )
    _libc_free((unsigned __int64)v39);
  return v76;
}
