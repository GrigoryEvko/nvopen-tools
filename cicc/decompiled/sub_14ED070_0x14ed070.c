// Function: sub_14ED070
// Address: 0x14ed070
//
__int64 __fastcall sub_14ED070(__int64 a1, char a2)
{
  unsigned int v3; // ebx
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // r9
  unsigned int v7; // r11d
  __int64 v8; // r14
  unsigned int v9; // r13d
  unsigned __int64 v10; // r8
  _QWORD *v11; // r10
  unsigned int v12; // edi
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // rax
  unsigned int v16; // edi
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdx
  char v21; // cl
  unsigned int v22; // edi
  unsigned __int64 v23; // r10
  unsigned __int64 v24; // r9
  __int64 v25; // r13
  unsigned int v26; // r14d
  unsigned __int64 v27; // r8
  unsigned __int64 *v28; // rbx
  unsigned __int64 v29; // rsi
  unsigned int v30; // r11d
  unsigned int v31; // eax
  unsigned __int64 v32; // r13
  __int64 v33; // r13
  unsigned int v34; // r14d
  unsigned __int64 *v35; // r9
  unsigned __int64 v36; // rdi
  unsigned int v37; // ebx
  unsigned __int64 v38; // rdx
  char v39; // cl
  __int64 v40; // r13
  unsigned int v41; // ebx
  unsigned __int64 *v42; // rdi
  unsigned __int64 v43; // r9
  unsigned int v44; // r14d
  unsigned int v45; // edi
  unsigned __int64 v46; // r13
  __int64 v47; // r13
  unsigned int v48; // r14d
  unsigned __int64 *v49; // r9
  unsigned __int64 v50; // rsi
  unsigned int v51; // ebx
  unsigned __int64 v52; // rax
  char v53; // cl
  unsigned __int64 v54; // rdx
  unsigned __int64 *v55; // rdi
  unsigned __int64 v56; // rax
  unsigned __int64 v57; // rbx
  unsigned __int64 v58; // rax
  unsigned __int64 *v59; // r9
  unsigned __int64 v60; // rdi
  unsigned int v61; // r10d
  unsigned int v62; // r10d
  unsigned __int64 v63; // rdi
  unsigned __int64 v64; // rax
  unsigned int v65; // ebx
  __int64 v66; // rax
  __int64 v67; // rdx
  char v68; // cl
  unsigned __int64 v69; // rdx
  unsigned __int64 v70; // rax
  unsigned __int64 v71; // rdx
  unsigned int v72; // r14d
  __int64 v73; // rdx
  __int64 v74; // rsi
  char v75; // cl
  unsigned int v76; // r9d
  __int64 v77; // rax
  unsigned __int64 v78; // rsi
  __int64 v79; // rdx
  char v80; // cl
  unsigned int v81; // r9d
  unsigned int v82; // r9d
  unsigned int v83; // r10d
  __int64 v84; // rdx
  __int64 v85; // rsi
  char v86; // cl
  unsigned __int64 v87; // rax
  unsigned __int64 v88; // rdx
  unsigned int v89; // ebx
  __int64 v90; // rdx
  __int64 v91; // rsi
  char v92; // cl
  unsigned int v93; // r11d
  __int64 v94; // rax
  __int64 v95; // rdx
  char v96; // cl
  __int64 v98; // [rsp+8h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 32);
LABEL_2:
  if ( v3 )
  {
    v7 = *(_DWORD *)(a1 + 36);
    if ( v7 <= v3 )
      goto LABEL_16;
    v4 = *(_QWORD *)(a1 + 8);
    v5 = *(_QWORD *)(a1 + 16);
    v8 = *(_QWORD *)(a1 + 24);
    v9 = v7 - v3;
    if ( v4 <= v5 )
      goto LABEL_75;
    goto LABEL_7;
  }
  while ( 1 )
  {
    v4 = *(_QWORD *)(a1 + 8);
    v5 = *(_QWORD *)(a1 + 16);
    if ( v4 <= v5 )
      return (v98 << 32) | v3;
    v9 = *(_DWORD *)(a1 + 36);
    if ( !v9 )
    {
      v7 = 0;
LABEL_16:
      v15 = *(_QWORD *)(a1 + 24);
      *(_DWORD *)(a1 + 32) = v3 - v7;
      v14 = v15 & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7));
      *(_QWORD *)(a1 + 24) = v15 >> v7;
      LODWORD(v98) = v14;
      if ( !(_DWORD)v14 )
        break;
      goto LABEL_11;
    }
    v7 = *(_DWORD *)(a1 + 36);
    v8 = 0;
LABEL_7:
    v10 = v5 + 8;
    v11 = (_QWORD *)(v5 + *(_QWORD *)a1);
    if ( v5 + 8 > v4 )
    {
      *(_QWORD *)(a1 + 24) = 0;
      v16 = v4 - v5;
      if ( !v16 )
        goto LABEL_108;
      v17 = v16;
      v18 = 0;
      v19 = 0;
      do
      {
        v20 = *((unsigned __int8 *)v11 + v18);
        v21 = 8 * v18++;
        v19 |= v20 << v21;
        *(_QWORD *)(a1 + 24) = v19;
      }
      while ( v16 != v18 );
      v12 = 8 * v16;
      v10 = v5 + v17;
    }
    else
    {
      v12 = 64;
      *(_QWORD *)(a1 + 24) = *v11;
    }
    *(_QWORD *)(a1 + 16) = v10;
    *(_DWORD *)(a1 + 32) = v12;
    if ( v9 > v12 )
      goto LABEL_75;
    v13 = *(_QWORD *)(a1 + 24);
    *(_DWORD *)(a1 + 32) = v3 - v7 + v12;
    *(_QWORD *)(a1 + 24) = v13 >> v9;
    v14 = ((v13 & (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v3 - (unsigned __int8)v7 + 64))) << v3) | v8;
    LODWORD(v98) = v14;
    if ( !(_DWORD)v14 )
      break;
LABEL_11:
    if ( (_DWORD)v14 != 1 )
    {
      if ( (_DWORD)v98 != 2 || (a2 & 2) != 0 )
      {
        v3 = 3;
        return (v98 << 32) | v3;
      }
      sub_1513230(a1);
      v3 = *(_DWORD *)(a1 + 32);
      goto LABEL_2;
    }
    v22 = *(_DWORD *)(a1 + 32);
    v23 = *(_QWORD *)(a1 + 8);
    v24 = *(_QWORD *)(a1 + 16);
    if ( v22 > 7 )
    {
      v27 = *(_QWORD *)(a1 + 16);
      LOBYTE(v32) = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 24) >>= 8;
      v31 = v22 - 8;
      *(_DWORD *)(a1 + 32) = v22 - 8;
    }
    else
    {
      v25 = 0;
      if ( v22 )
        v25 = *(_QWORD *)(a1 + 24);
      v26 = 8 - v22;
      if ( v23 <= v24 )
        goto LABEL_75;
      v27 = v24 + 8;
      v28 = (unsigned __int64 *)(v24 + *(_QWORD *)a1);
      if ( v24 + 8 > v23 )
      {
        *(_QWORD *)(a1 + 24) = 0;
        v93 = v23 - v24;
        if ( (_DWORD)v23 == (_DWORD)v24 )
        {
LABEL_108:
          *(_DWORD *)(a1 + 32) = 0;
          goto LABEL_75;
        }
        v94 = 0;
        v29 = 0;
        do
        {
          v95 = *((unsigned __int8 *)v28 + v94);
          v96 = 8 * v94++;
          v29 |= v95 << v96;
          *(_QWORD *)(a1 + 24) = v29;
        }
        while ( v93 != v94 );
        v27 = v24 + v93;
        v30 = 8 * v93;
        *(_QWORD *)(a1 + 16) = v27;
        *(_DWORD *)(a1 + 32) = v30;
        if ( v26 > v30 )
          goto LABEL_75;
      }
      else
      {
        v29 = *v28;
        *(_QWORD *)(a1 + 16) = v27;
        v30 = 64;
      }
      *(_QWORD *)(a1 + 24) = v29 >> v26;
      v31 = v22 + v30 - 8;
      *(_DWORD *)(a1 + 32) = v31;
      v32 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v22 + 56)) & v29) << v22) | v25;
    }
    while ( (v32 & 0x80) != 0 )
    {
      while ( v31 <= 7 )
      {
        LOBYTE(v33) = 0;
        if ( v31 )
          v33 = *(_QWORD *)(a1 + 24);
        v34 = 8 - v31;
        if ( v27 >= v23 )
          goto LABEL_75;
        v35 = (unsigned __int64 *)(v27 + *(_QWORD *)a1);
        if ( v27 + 8 > v23 )
        {
          *(_QWORD *)(a1 + 24) = 0;
          v89 = v23 - v27;
          if ( (_DWORD)v23 == (_DWORD)v27 )
            goto LABEL_107;
          v90 = 0;
          v36 = 0;
          do
          {
            v91 = *((unsigned __int8 *)v35 + v90);
            v92 = 8 * v90++;
            v36 |= v91 << v92;
            *(_QWORD *)(a1 + 24) = v36;
          }
          while ( v89 != v90 );
          v27 += v89;
          v37 = 8 * v89;
          *(_QWORD *)(a1 + 16) = v27;
          *(_DWORD *)(a1 + 32) = v37;
          if ( v34 > v37 )
            goto LABEL_75;
        }
        else
        {
          v36 = *v35;
          *(_QWORD *)(a1 + 16) = v27 + 8;
          v27 += 8LL;
          v37 = 64;
        }
        *(_DWORD *)(a1 + 32) = v37 + v31 - 8;
        *(_QWORD *)(a1 + 24) = v36 >> v34;
        v38 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v31 + 56);
        v39 = v31;
        v31 = v37 + v31 - 8;
        if ( (((unsigned __int8)((v38 & v36) << v39) | (unsigned __int8)v33) & 0x80) == 0 )
          goto LABEL_43;
      }
      v88 = *(_QWORD *)(a1 + 24);
      v31 -= 8;
      *(_DWORD *)(a1 + 32) = v31;
      LOBYTE(v32) = v88;
      *(_QWORD *)(a1 + 24) = v88 >> 8;
    }
LABEL_43:
    if ( v31 > 3 )
    {
      v69 = *(_QWORD *)(a1 + 24);
      v45 = v31 - 4;
      *(_DWORD *)(a1 + 32) = v31 - 4;
      *(_QWORD *)(a1 + 24) = v69 >> 4;
      LOBYTE(v46) = v69 & 0xF;
    }
    else
    {
      v40 = 0;
      if ( v31 )
        v40 = *(_QWORD *)(a1 + 24);
      v41 = 4 - v31;
      if ( v27 >= v23 )
        goto LABEL_75;
      v42 = (unsigned __int64 *)(v27 + *(_QWORD *)a1);
      if ( v27 + 8 > v23 )
      {
        *(_QWORD *)(a1 + 24) = 0;
        v72 = v23 - v27;
        if ( (_DWORD)v23 == (_DWORD)v27 )
        {
LABEL_107:
          *(_QWORD *)(a1 + 16) = v27;
          *(_DWORD *)(a1 + 32) = 0;
          goto LABEL_75;
        }
        v73 = 0;
        v43 = 0;
        do
        {
          v74 = *((unsigned __int8 *)v42 + v73);
          v75 = 8 * v73++;
          v43 |= v74 << v75;
          *(_QWORD *)(a1 + 24) = v43;
        }
        while ( v72 != v73 );
        v27 += v72;
        v44 = 8 * v72;
        *(_QWORD *)(a1 + 16) = v27;
        *(_DWORD *)(a1 + 32) = v44;
        if ( v41 > v44 )
          goto LABEL_75;
      }
      else
      {
        v43 = *v42;
        *(_QWORD *)(a1 + 16) = v27 + 8;
        v27 += 8LL;
        v44 = 64;
      }
      v45 = v31 + v44 - 4;
      *(_DWORD *)(a1 + 32) = v45;
      *(_QWORD *)(a1 + 24) = v43 >> v41;
      v46 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v31 + 60)) & v43) << v31) | v40;
    }
    while ( (v46 & 8) != 0 )
    {
      while ( v45 <= 3 )
      {
        LOBYTE(v47) = 0;
        if ( v45 )
          v47 = *(_QWORD *)(a1 + 24);
        v48 = 4 - v45;
        if ( v27 >= v23 )
          goto LABEL_75;
        v49 = (unsigned __int64 *)(v27 + *(_QWORD *)a1);
        if ( v27 + 8 > v23 )
        {
          *(_QWORD *)(a1 + 24) = 0;
          v65 = v23 - v27;
          if ( (_DWORD)v23 == (_DWORD)v27 )
            goto LABEL_107;
          v66 = 0;
          v50 = 0;
          do
          {
            v67 = *((unsigned __int8 *)v49 + v66);
            v68 = 8 * v66++;
            v50 |= v67 << v68;
            *(_QWORD *)(a1 + 24) = v50;
          }
          while ( v65 != v66 );
          v27 += v65;
          v51 = 8 * v65;
          *(_QWORD *)(a1 + 16) = v27;
          *(_DWORD *)(a1 + 32) = v51;
          if ( v48 > v51 )
            goto LABEL_75;
        }
        else
        {
          v50 = *v49;
          *(_QWORD *)(a1 + 16) = v27 + 8;
          v27 += 8LL;
          v51 = 64;
        }
        *(_DWORD *)(a1 + 32) = v45 + v51 - 4;
        *(_QWORD *)(a1 + 24) = v50 >> v48;
        v52 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v45 + 60);
        v53 = v45;
        v45 = v45 + v51 - 4;
        if ( (((unsigned __int8)((v52 & v50) << v53) | (unsigned __int8)v47) & 8) == 0 )
          goto LABEL_58;
      }
      v64 = *(_QWORD *)(a1 + 24);
      v45 -= 4;
      *(_DWORD *)(a1 + 32) = v45;
      *(_QWORD *)(a1 + 24) = v64 >> 4;
      LOBYTE(v46) = v64 & 0xF;
    }
LABEL_58:
    if ( v45 > 0x1F )
    {
      v70 = *(_QWORD *)(a1 + 24);
      *(_DWORD *)(a1 + 32) = 0;
      v71 = v70 >> ((unsigned __int8)v45 - 32);
      *(_QWORD *)(a1 + 24) = HIDWORD(v71);
      v57 = 32LL * (unsigned int)v71 + 8 * v27;
LABEL_78:
      if ( v27 >= v23 )
        return 0;
      goto LABEL_62;
    }
    *(_DWORD *)(a1 + 32) = 0;
    if ( v27 >= v23 )
      goto LABEL_75;
    v54 = v27 + 8;
    v55 = (unsigned __int64 *)(v27 + *(_QWORD *)a1);
    if ( v27 + 8 <= v23 )
    {
      v56 = *v55;
      *(_QWORD *)(a1 + 16) = v54;
      *(_DWORD *)(a1 + 32) = 32;
      *(_QWORD *)(a1 + 24) = HIDWORD(v56);
      v57 = 32LL * (unsigned int)v56 + 8 * v54 - 32;
      goto LABEL_62;
    }
    *(_QWORD *)(a1 + 24) = 0;
    v76 = v23 - v27;
    if ( (_DWORD)v23 == (_DWORD)v27 )
    {
      *(_QWORD *)(a1 + 16) = v27;
LABEL_75:
      sub_16BD130("Unexpected end of file", 1);
    }
    v77 = 0;
    v78 = 0;
    do
    {
      v79 = *((unsigned __int8 *)v55 + v77);
      v80 = 8 * v77++;
      v78 |= v79 << v80;
      *(_QWORD *)(a1 + 24) = v78;
    }
    while ( v76 != v77 );
    v27 += v76;
    v81 = 8 * v76;
    *(_QWORD *)(a1 + 16) = v27;
    *(_DWORD *)(a1 + 32) = v81;
    if ( v81 <= 0x1F )
      goto LABEL_75;
    v82 = v81 - 32;
    *(_DWORD *)(a1 + 32) = v82;
    *(_QWORD *)(a1 + 24) = HIDWORD(v78);
    v57 = 32LL * (unsigned int)v78 + 8 * v27 - v82;
    if ( !v82 )
      goto LABEL_78;
LABEL_62:
    if ( v57 >> 3 > v23 )
      return 0;
    *(_DWORD *)(a1 + 32) = 0;
    v58 = (v57 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 16) = v58;
    v3 = v57 & 0x3F;
    if ( v3 )
    {
      if ( v58 >= v23 )
        goto LABEL_75;
      v59 = (unsigned __int64 *)(v58 + *(_QWORD *)a1);
      if ( v58 + 8 > v23 )
      {
        *(_QWORD *)(a1 + 24) = 0;
        v83 = v23 - v58;
        if ( !v83 )
          goto LABEL_75;
        v84 = 0;
        v60 = 0;
        do
        {
          v85 = *((unsigned __int8 *)v59 + v84);
          v86 = 8 * v84++;
          v60 |= v85 << v86;
          *(_QWORD *)(a1 + 24) = v60;
        }
        while ( v83 != v84 );
        v87 = v83 + v58;
        v61 = 8 * v83;
        *(_QWORD *)(a1 + 16) = v87;
        *(_DWORD *)(a1 + 32) = v61;
        if ( v3 > v61 )
          goto LABEL_75;
      }
      else
      {
        v60 = *v59;
        *(_QWORD *)(a1 + 16) = v58 + 8;
        v61 = 64;
      }
      v62 = v61 - v3;
      v63 = v60 >> v3;
      *(_DWORD *)(a1 + 32) = v62;
      v3 = v62;
      *(_QWORD *)(a1 + 24) = v63;
      goto LABEL_2;
    }
  }
  v3 = a2 & 1;
  if ( (a2 & 1) != 0 || *(_DWORD *)(a1 + 72) && !(unsigned __int8)sub_14EB5C0(a1) )
    v3 = 1;
  return (v98 << 32) | v3;
}
