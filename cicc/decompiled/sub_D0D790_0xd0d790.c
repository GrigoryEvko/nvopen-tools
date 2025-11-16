// Function: sub_D0D790
// Address: 0xd0d790
//
__int64 __fastcall sub_D0D790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  _QWORD *v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // rsi
  __int64 v11; // rdx
  _QWORD *v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // rdx
  bool v15; // zf
  __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 *v18; // rdx
  __int64 v19; // r14
  __int64 *v20; // rax
  __int64 v21; // rsi
  unsigned int v22; // r12d
  __int64 *v24; // rax
  __int64 *v25; // rcx
  __int64 v26; // rsi
  __int64 *i; // r8
  __int64 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rsi
  __int64 *v31; // rcx
  _QWORD *v32; // rax
  __int64 *v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rsi
  __int64 *v36; // rax
  __int64 *v37; // rax
  char v38; // dl
  _QWORD *v39; // rax
  _QWORD *v40; // rdx
  _QWORD *v41; // rax
  _QWORD *v42; // rdx
  __int64 *v43; // r13
  __int64 v44; // rax
  __int64 *v45; // r15
  __int64 v46; // rdx
  _QWORD *v47; // rax
  __int64 *v48; // rax
  __int64 *v49; // rdx
  __int64 *v50; // rax
  __int64 *v51; // rdx
  __int64 *v52; // rax
  unsigned __int64 v53; // rax
  __int64 v54; // r15
  int v55; // r14d
  __int64 v56; // r13
  __int64 v57; // rdx
  __int64 v58; // rcx
  unsigned int j; // edx
  __int64 v60; // rax
  unsigned int v61; // edx
  __int64 *v62; // rax
  __int64 *v63; // rax
  unsigned int v64; // edi
  __int64 v65; // rdx
  _QWORD *v66; // rax
  _QWORD *v67; // rax
  __int64 *v68; // rdx
  __int64 v69; // r8
  __int64 v70; // rsi
  __int64 *v71; // rax
  __int64 *v72; // rax
  int v73; // [rsp+14h] [rbp-20Ch]
  __int64 v75; // [rsp+20h] [rbp-200h]
  __int64 v77; // [rsp+30h] [rbp-1F0h]
  _QWORD *v78; // [rsp+30h] [rbp-1F0h]
  __int64 v79; // [rsp+30h] [rbp-1F0h]
  __int64 v80; // [rsp+38h] [rbp-1E8h]
  unsigned int v81; // [rsp+38h] [rbp-1E8h]
  __int64 v82; // [rsp+38h] [rbp-1E8h]
  _QWORD *v83; // [rsp+38h] [rbp-1E8h]
  __int64 v84; // [rsp+38h] [rbp-1E8h]
  __int64 v85; // [rsp+40h] [rbp-1E0h] BYREF
  __int64 *v86; // [rsp+48h] [rbp-1D8h]
  __int64 v87; // [rsp+50h] [rbp-1D0h]
  int v88; // [rsp+58h] [rbp-1C8h]
  char v89; // [rsp+5Ch] [rbp-1C4h]
  _BYTE v90[16]; // [rsp+60h] [rbp-1C0h] BYREF
  __int64 v91; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 *v92; // [rsp+78h] [rbp-1A8h]
  __int64 v93; // [rsp+80h] [rbp-1A0h]
  int v94; // [rsp+88h] [rbp-198h]
  char v95; // [rsp+8Ch] [rbp-194h]
  _BYTE v96[64]; // [rsp+90h] [rbp-190h] BYREF
  __int64 v97; // [rsp+D0h] [rbp-150h] BYREF
  __int64 *v98; // [rsp+D8h] [rbp-148h]
  __int64 v99; // [rsp+E0h] [rbp-140h]
  int v100; // [rsp+E8h] [rbp-138h]
  unsigned __int8 v101; // [rsp+ECh] [rbp-134h]
  char v102; // [rsp+F0h] [rbp-130h] BYREF

  v6 = a4;
  v75 = a5;
  if ( !a4 )
    goto LABEL_7;
  v8 = *(_QWORD **)(a2 + 8);
  v9 = *(_BYTE *)(a2 + 28) ? *(unsigned int *)(a2 + 20) : *(unsigned int *)(a2 + 16);
  v10 = &v8[v9];
  if ( v8 == v10 )
    goto LABEL_7;
  while ( 1 )
  {
    v11 = *v8;
    v12 = v8;
    if ( *v8 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v10 == ++v8 )
      goto LABEL_7;
  }
  if ( v10 == v8 )
  {
LABEL_7:
    v13 = a3;
    if ( a3 )
      goto LABEL_8;
LABEL_137:
    v91 = 0;
    v92 = (__int64 *)v96;
    v93 = 8;
    v94 = 0;
    v95 = 1;
    v85 = 0;
    v86 = (__int64 *)v90;
    v87 = 2;
    v88 = 0;
    v89 = 1;
    if ( !a5 )
      goto LABEL_12;
LABEL_30:
    v28 = *(__int64 **)(a2 + 8);
    if ( *(_BYTE *)(a2 + 28) )
      v29 = *(unsigned int *)(a2 + 20);
    else
      v29 = *(unsigned int *)(a2 + 16);
    a5 = (__int64)&v28[v29];
    if ( v28 == (__int64 *)a5 )
      goto LABEL_12;
    while ( 1 )
    {
      v30 = *v28;
      v31 = v28;
      if ( (unsigned __int64)*v28 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( (__int64 *)a5 == ++v28 )
        goto LABEL_12;
    }
    if ( v28 == (__int64 *)a5 )
      goto LABEL_12;
    while ( 1 )
    {
      v77 = a5;
      v80 = (__int64)v31;
      v32 = sub_D0CF30(v75, v30);
      v34 = v80;
      a5 = v77;
      v35 = (__int64)v32;
      if ( v32 )
      {
        if ( !v89 )
          goto LABEL_121;
        v36 = v86;
        v33 = &v86[HIDWORD(v87)];
        if ( v86 != v33 )
        {
          while ( v35 != *v36 )
          {
            if ( v33 == ++v36 )
              goto LABEL_122;
          }
          goto LABEL_43;
        }
LABEL_122:
        if ( HIDWORD(v87) < (unsigned int)v87 )
        {
          ++HIDWORD(v87);
          *v33 = v35;
          ++v85;
        }
        else
        {
LABEL_121:
          sub_C8CC70((__int64)&v85, v35, (__int64)v33, v80, v77, a6);
          a5 = v77;
          v34 = v80;
        }
      }
LABEL_43:
      v37 = (__int64 *)(v34 + 8);
      if ( v34 + 8 != a5 )
      {
        while ( 1 )
        {
          v30 = *v37;
          v31 = v37;
          if ( (unsigned __int64)*v37 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( (__int64 *)a5 == ++v37 )
            goto LABEL_12;
        }
        if ( v37 != (__int64 *)a5 )
          continue;
      }
      goto LABEL_12;
    }
  }
  v64 = *(_DWORD *)(v6 + 32);
  if ( !v11 )
    goto LABEL_135;
LABEL_127:
  v65 = (unsigned int)(*(_DWORD *)(v11 + 44) + 1);
  if ( (unsigned int)v65 < v64 )
  {
    while ( 1 )
    {
      if ( !*(_QWORD *)(*(_QWORD *)(v6 + 24) + 8 * v65) )
        goto LABEL_136;
      v66 = v12 + 1;
      if ( v12 + 1 == v10 )
        goto LABEL_7;
      v11 = *v66;
      ++v12;
      if ( *v66 >= 0xFFFFFFFFFFFFFFFELL )
        break;
LABEL_133:
      if ( v10 == v12 )
        goto LABEL_7;
      if ( v11 )
        goto LABEL_127;
LABEL_135:
      v65 = 0;
      if ( !v64 )
        goto LABEL_136;
    }
    while ( 1 )
    {
      if ( v10 == ++v66 )
        goto LABEL_7;
      v11 = *v66;
      v12 = v66;
      if ( *v66 < 0xFFFFFFFFFFFFFFFELL )
        goto LABEL_133;
    }
  }
LABEL_136:
  v13 = a3;
  v6 = 0;
  if ( !a3 )
    goto LABEL_137;
LABEL_8:
  v14 = *(unsigned int *)(v13 + 20);
  v15 = *(_DWORD *)(v13 + 24) == (_DWORD)v14;
  v91 = 0;
  if ( !v15 )
    v6 = 0;
  v93 = 8;
  v92 = (__int64 *)v96;
  v94 = 0;
  v95 = 1;
  if ( a5 )
  {
    v24 = *(__int64 **)(v13 + 8);
    if ( !*(_BYTE *)(v13 + 28) )
      v14 = *(unsigned int *)(a3 + 16);
    v25 = &v24[v14];
    if ( v24 != v25 )
    {
      while ( 1 )
      {
        v26 = *v24;
        i = v24;
        if ( (unsigned __int64)*v24 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v25 == ++v24 )
          goto LABEL_29;
      }
      if ( v25 != v24 )
      {
        while ( 1 )
        {
          v79 = (__int64)v25;
          v84 = (__int64)i;
          v67 = sub_D0CF30(v75, v26);
          v69 = v84;
          v25 = (__int64 *)v79;
          v70 = (__int64)v67;
          if ( !v67 )
            goto LABEL_146;
          if ( !v95 )
            break;
          v71 = v92;
          v68 = &v92[HIDWORD(v93)];
          if ( v92 == v68 )
          {
LABEL_153:
            if ( HIDWORD(v93) >= (unsigned int)v93 )
              break;
            ++HIDWORD(v93);
            *v68 = v70;
            ++v91;
          }
          else
          {
            while ( v70 != *v71 )
            {
              if ( v68 == ++v71 )
                goto LABEL_153;
            }
          }
LABEL_146:
          v72 = (__int64 *)(v69 + 8);
          if ( (__int64 *)(v69 + 8) == v25 )
            goto LABEL_29;
          v26 = *v72;
          for ( i = (__int64 *)(v69 + 8); (unsigned __int64)*v72 >= 0xFFFFFFFFFFFFFFFELL; i = v72 )
          {
            if ( v25 == ++v72 )
              goto LABEL_29;
            v26 = *v72;
          }
          if ( v25 == i )
            goto LABEL_29;
        }
        sub_C8CC70((__int64)&v91, v70, (__int64)v68, v79, v84, a6);
        v25 = (__int64 *)v79;
        v69 = v84;
        goto LABEL_146;
      }
    }
LABEL_29:
    v85 = 0;
    v86 = (__int64 *)v90;
    v87 = 2;
    v88 = 0;
    v89 = 1;
    goto LABEL_30;
  }
  v85 = 0;
  v86 = (__int64 *)v90;
  v87 = 2;
  v88 = 0;
  v89 = 1;
LABEL_12:
  v97 = 0;
  v16 = 1;
  v99 = 32;
  v73 = qword_4F86A48;
  v100 = 0;
  v101 = 1;
  v98 = (__int64 *)&v102;
  v17 = *(_DWORD *)(a1 + 8);
  while ( 1 )
  {
    v18 = *(__int64 **)a1;
    v19 = *(_QWORD *)(*(_QWORD *)a1 + 8LL * v17 - 8);
    *(_DWORD *)(a1 + 8) = v17 - 1;
    if ( !(_BYTE)v16 )
      goto LABEL_50;
    v20 = v98;
    v21 = HIDWORD(v99);
    v18 = &v98[HIDWORD(v99)];
    if ( v98 != v18 )
    {
      while ( v19 != *v20 )
      {
        if ( v18 == ++v20 )
          goto LABEL_109;
      }
      goto LABEL_18;
    }
LABEL_109:
    if ( HIDWORD(v99) < (unsigned int)v99 )
    {
      v21 = (unsigned int)++HIDWORD(v99);
      *v18 = v19;
      LOBYTE(v16) = v101;
      ++v97;
    }
    else
    {
LABEL_50:
      v21 = v19;
      sub_C8CC70((__int64)&v97, v19, (__int64)v18, v16, a5, a6);
      v16 = v101;
      if ( !v38 )
        goto LABEL_18;
    }
    if ( *(_BYTE *)(a2 + 28) )
    {
      v39 = *(_QWORD **)(a2 + 8);
      v40 = &v39[*(unsigned int *)(a2 + 20)];
      if ( v39 != v40 )
      {
        while ( v19 != *v39 )
        {
          if ( v40 == ++v39 )
            goto LABEL_62;
        }
        v22 = 1;
        goto LABEL_57;
      }
    }
    else
    {
      v21 = v19;
      if ( sub_C8CA60(a2, v19) )
        goto LABEL_88;
    }
LABEL_62:
    if ( !a3 )
      break;
    if ( *(_BYTE *)(a3 + 28) )
    {
      v41 = *(_QWORD **)(a3 + 8);
      v42 = &v41[*(unsigned int *)(a3 + 20)];
      if ( v41 == v42 )
        break;
      while ( v19 != *v41 )
      {
        if ( v42 == ++v41 )
          goto LABEL_70;
      }
    }
    else
    {
      v21 = v19;
      if ( !sub_C8CA60(a3, v19) )
        break;
    }
    v16 = v101;
LABEL_18:
    v17 = *(_DWORD *)(a1 + 8);
LABEL_19:
    if ( !v17 )
    {
      v22 = 0;
      if ( !(_BYTE)v16 )
        goto LABEL_58;
LABEL_21:
      if ( !v89 )
        goto LABEL_59;
LABEL_22:
      if ( v95 )
        return v22;
      goto LABEL_60;
    }
  }
LABEL_70:
  if ( v6 )
  {
    v43 = *(__int64 **)(a2 + 8);
    v44 = *(_BYTE *)(a2 + 28) ? *(unsigned int *)(a2 + 20) : *(unsigned int *)(a2 + 16);
    v45 = &v43[v44];
    if ( v43 != v45 )
    {
      while ( 1 )
      {
        v46 = *v43;
        if ( (unsigned __int64)*v43 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( ++v43 == v45 )
          goto LABEL_76;
      }
      while ( v43 != v45 )
      {
        v21 = v19;
        if ( (unsigned __int8)sub_B19720(v6, v19, v46) )
        {
          if ( v43 == v45 )
            break;
          goto LABEL_88;
        }
        v52 = v43 + 1;
        if ( v43 + 1 == v45 )
          break;
        while ( 1 )
        {
          v46 = *v52;
          v43 = v52;
          if ( (unsigned __int64)*v52 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v45 == ++v52 )
          {
            if ( v75 )
              goto LABEL_77;
            goto LABEL_95;
          }
        }
      }
    }
  }
LABEL_76:
  if ( !v75 )
  {
LABEL_95:
    if ( !--v73 )
      goto LABEL_88;
    goto LABEL_96;
  }
LABEL_77:
  v21 = v19;
  v47 = sub_D0CF30(v75, v19);
  a6 = (__int64)v47;
  if ( v95 )
  {
    v48 = v92;
    v49 = &v92[HIDWORD(v93)];
    if ( v92 == v49 )
      goto LABEL_83;
    while ( a6 != *v48 )
    {
      if ( v49 == ++v48 )
        goto LABEL_83;
    }
LABEL_82:
    a6 = 0;
    goto LABEL_83;
  }
  v21 = (__int64)v47;
  v83 = v47;
  v63 = sub_C8CA60((__int64)&v91, (__int64)v47);
  a6 = (__int64)v83;
  if ( v63 )
    goto LABEL_82;
LABEL_83:
  if ( !v89 )
  {
    v21 = a6;
    v82 = a6;
    v62 = sub_C8CA60((__int64)&v85, a6);
    a6 = v82;
    if ( v62 )
      goto LABEL_88;
LABEL_112:
    if ( !--v73 )
      goto LABEL_88;
    if ( a6 )
    {
      v21 = a1;
      sub_D472F0(a6, a1);
      v17 = *(_DWORD *)(a1 + 8);
LABEL_115:
      v16 = v101;
      goto LABEL_19;
    }
LABEL_96:
    v53 = *(_QWORD *)(v19 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v53 == v19 + 48 )
      goto LABEL_124;
    if ( !v53 )
      BUG();
    v54 = v53 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v53 - 24) - 30 > 0xA )
    {
LABEL_124:
      v56 = 0;
      v55 = 0;
      v54 = 0;
    }
    else
    {
      v55 = sub_B46E30(v53 - 24);
      v56 = v55;
    }
    v57 = *(unsigned int *)(a1 + 8);
    a5 = v56 + v57;
    if ( v56 + v57 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      v21 = a1 + 16;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v56 + v57, 8u, a5, a6);
      v57 = *(unsigned int *)(a1 + 8);
    }
    v58 = *(_QWORD *)a1 + 8 * v57;
    if ( v55 )
    {
      for ( j = 0; j != v55; ++j )
      {
        if ( v58 )
        {
          v21 = j;
          v78 = (_QWORD *)v58;
          v81 = j;
          v60 = sub_B46EC0(v54, j);
          v58 = (__int64)v78;
          j = v81;
          *v78 = v60;
        }
        v58 += 8;
      }
      LODWORD(v57) = *(_DWORD *)(a1 + 8);
    }
    v61 = v56 + v57;
    *(_DWORD *)(a1 + 8) = v61;
    v17 = v61;
    goto LABEL_115;
  }
  v50 = v86;
  v51 = &v86[HIDWORD(v87)];
  if ( v86 == v51 )
    goto LABEL_112;
  while ( a6 != *v50 )
  {
    if ( v51 == ++v50 )
      goto LABEL_112;
  }
LABEL_88:
  LOBYTE(v16) = v101;
  v22 = 1;
LABEL_57:
  if ( (_BYTE)v16 )
    goto LABEL_21;
LABEL_58:
  _libc_free(v98, v21);
  if ( v89 )
    goto LABEL_22;
LABEL_59:
  _libc_free(v86, v21);
  if ( v95 )
    return v22;
LABEL_60:
  _libc_free(v92, v21);
  return v22;
}
