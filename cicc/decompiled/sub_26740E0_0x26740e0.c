// Function: sub_26740E0
// Address: 0x26740e0
//
__int64 __fastcall sub_26740E0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r8
  __int64 v3; // r13
  __int64 v6; // r9
  int v7; // eax
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned int v11; // eax
  __int64 *v12; // r13
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned int v17; // r13d
  int v18; // edx
  _QWORD *v19; // rdi
  _QWORD *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rsi
  unsigned int v23; // ecx
  __int64 *v24; // rdx
  __int64 v25; // r10
  __int64 v26; // r14
  unsigned __int64 i; // r8
  __int64 *v28; // rcx
  __int64 v29; // r9
  __int64 result; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rdx
  int v35; // eax
  __int64 v36; // rcx
  __int64 v37; // rsi
  int v38; // edx
  unsigned int v39; // eax
  __int64 *v40; // r15
  __int64 v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  char v45; // r15
  int v46; // eax
  _QWORD *v47; // rdi
  _QWORD *v48; // rsi
  __int64 v49; // rax
  __int64 v50; // rsi
  unsigned int v51; // ecx
  __int64 *v52; // rdx
  __int64 v53; // r10
  __int64 v54; // rsi
  __int64 v55; // rdx
  __int64 *v56; // rdi
  int v57; // ecx
  unsigned int v58; // eax
  __int64 *v59; // rdx
  __int64 v60; // r9
  __int64 v61; // rsi
  __int64 v62; // rdx
  __int64 *v63; // rdi
  int v64; // ecx
  unsigned int v65; // eax
  __int64 *v66; // rdx
  __int64 v67; // r9
  int v68; // edx
  int v69; // r11d
  int v70; // edx
  int v71; // r11d
  int v72; // edx
  int v73; // r10d
  int v74; // eax
  __int64 *v75; // rbx
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  unsigned __int8 v80; // al
  int v81; // edx
  int v82; // r10d
  int v83; // eax
  int v84; // ecx
  unsigned int v85; // esi
  int v86; // edi
  unsigned __int8 v87; // [rsp+Fh] [rbp-B1h]
  __int64 v88[22]; // [rsp+10h] [rbp-B0h] BYREF

  v2 = a2;
  v3 = 0x8000000000041LL;
  do
  {
    if ( (unsigned __int8)(*(_BYTE *)v2 - 34) <= 0x33u && _bittest64(&v3, (unsigned int)*(unsigned __int8 *)v2 - 34) )
    {
      if ( v2 != a2 )
      {
        v18 = *(_DWORD *)(a1 + 304);
        v88[0] = v2;
        if ( v18 )
        {
          v54 = *(_QWORD *)(a1 + 296);
          v55 = *(unsigned int *)(a1 + 312);
          v56 = (__int64 *)(v54 + 8 * v55);
          if ( (_DWORD)v55 )
          {
            v57 = v55 - 1;
            v58 = (v55 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
            v59 = (__int64 *)(v54 + 8LL * v58);
            v60 = *v59;
            if ( v2 == *v59 )
            {
LABEL_49:
              if ( v56 != v59 )
                return 1;
            }
            else
            {
              v81 = 1;
              while ( v60 != -4096 )
              {
                v82 = v81 + 1;
                v58 = v57 & (v81 + v58);
                v59 = (__int64 *)(v54 + 8LL * v58);
                v60 = *v59;
                if ( v2 == *v59 )
                  goto LABEL_49;
                v81 = v82;
              }
            }
          }
        }
        else
        {
          v19 = *(_QWORD **)(a1 + 320);
          v20 = &v19[*(unsigned int *)(a1 + 328)];
          if ( v20 != sub_266E350(v19, (__int64)v20, v88) )
            return 1;
        }
      }
      v21 = *(unsigned int *)(a1 + 280);
      if ( (_DWORD)v21 )
      {
        v22 = *(_QWORD *)(a1 + 264);
        v23 = (v21 - 1) & (v2 & 0xFFFFFFFB ^ (v2 >> 9));
        v24 = (__int64 *)(v22 + ((unsigned __int64)v23 << 7));
        v25 = *v24;
        if ( (v2 & 0xFFFFFFFFFFFFFFFBLL) == *v24 )
        {
LABEL_16:
          if ( v24 != (__int64 *)(v22 + (v21 << 7)) )
          {
            v17 = *((unsigned __int8 *)v24 + 10);
            goto LABEL_18;
          }
        }
        else
        {
          v68 = 1;
          while ( v25 != -4 )
          {
            v69 = v68 + 1;
            v23 = (v21 - 1) & (v68 + v23);
            v24 = (__int64 *)(v22 + ((unsigned __int64)v23 << 7));
            v25 = *v24;
            if ( (v2 & 0xFFFFFFFFFFFFFFFBLL) == *v24 )
              goto LABEL_16;
            v68 = v69;
          }
        }
      }
    }
    v2 = sub_B46B10(v2, 0);
  }
  while ( v2 );
  v7 = *(_DWORD *)(a1 + 248);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)(a1 + 232);
  if ( v7 )
  {
    v10 = (unsigned int)(v7 - 1);
    v11 = v10 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v12 = (__int64 *)(v9 + ((unsigned __int64)v11 << 7));
    v13 = *v12;
    if ( v8 == *v12 )
    {
LABEL_6:
      LODWORD(v88[0]) = *((_DWORD *)v12 + 2);
      sub_C8CD80((__int64)&v88[1], (__int64)&v88[5], (__int64)(v12 + 2), v10, v2, v6);
      sub_C8CD80((__int64)&v88[7], (__int64)&v88[11], (__int64)(v12 + 8), v14, v15, v16);
      v17 = BYTE2(v88[0]);
      if ( !BYTE4(v88[10]) )
        _libc_free(v88[8]);
      if ( !BYTE4(v88[4]) )
        _libc_free(v88[2]);
      goto LABEL_18;
    }
    v2 = 1;
    while ( v13 != -4096 )
    {
      v6 = (unsigned int)(v2 + 1);
      v11 = v10 & (v2 + v11);
      v12 = (__int64 *)(v9 + ((unsigned __int64)v11 << 7));
      v13 = *v12;
      if ( v8 == *v12 )
        goto LABEL_6;
      v2 = (unsigned int)v6;
    }
  }
  v17 = 1;
LABEL_18:
  v26 = 0x8000000000041LL;
  i = a2;
  do
  {
    if ( (unsigned __int8)(*(_BYTE *)i - 34) <= 0x33u && _bittest64(&v26, (unsigned int)*(unsigned __int8 *)i - 34) )
    {
      if ( a2 != i )
      {
        v46 = *(_DWORD *)(a1 + 304);
        v88[0] = i;
        if ( v46 )
        {
          v61 = *(_QWORD *)(a1 + 296);
          v62 = *(unsigned int *)(a1 + 312);
          v63 = (__int64 *)(v61 + 8 * v62);
          if ( (_DWORD)v62 )
          {
            v64 = v62 - 1;
            v65 = (v62 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
            v66 = (__int64 *)(v61 + 8LL * v65);
            v67 = *v66;
            if ( i == *v66 )
            {
LABEL_53:
              if ( v63 != v66 )
                return 1;
            }
            else
            {
              v72 = 1;
              while ( v67 != -4096 )
              {
                v73 = v72 + 1;
                v65 = v64 & (v72 + v65);
                v66 = (__int64 *)(v61 + 8LL * v65);
                v67 = *v66;
                if ( i == *v66 )
                  goto LABEL_53;
                v72 = v73;
              }
            }
          }
        }
        else
        {
          v47 = *(_QWORD **)(a1 + 320);
          v48 = &v47[*(unsigned int *)(a1 + 328)];
          if ( v48 != sub_266E350(v47, (__int64)v48, v88) )
            return 1;
        }
      }
      v49 = *(unsigned int *)(a1 + 280);
      if ( (_DWORD)v49 )
      {
        v50 = *(_QWORD *)(a1 + 264);
        v51 = (v49 - 1) & ((i | 4) ^ (i >> 9));
        v52 = (__int64 *)(v50 + ((unsigned __int64)v51 << 7));
        v53 = *v52;
        if ( (i | 4) == *v52 )
        {
LABEL_43:
          if ( v52 != (__int64 *)(v50 + (v49 << 7)) )
          {
            result = *((unsigned __int8 *)v52 + 9);
            if ( (_BYTE)result )
              return v17;
            return result;
          }
        }
        else
        {
          v70 = 1;
          while ( v53 != -4 )
          {
            v71 = v70 + 1;
            v51 = (v49 - 1) & (v70 + v51);
            v52 = (__int64 *)(v50 + ((unsigned __int64)v51 << 7));
            v53 = *v52;
            if ( (i | 4) == *v52 )
              goto LABEL_43;
            v70 = v71;
          }
        }
      }
    }
    i = sub_B46BC0(i, 0);
  }
  while ( i );
  result = 0;
  if ( !(_BYTE)v17 )
    return result;
  v31 = *(_QWORD *)(a2 + 40);
  v32 = *(_QWORD *)(*(_QWORD *)(v31 + 72) + 80LL);
  if ( v32 && v31 == v32 - 24 )
  {
    v74 = *(_DWORD *)(a1 + 248);
    v75 = *(__int64 **)(a1 + 232);
    if ( v74 )
    {
      v76 = *v75;
      if ( !*v75 )
      {
LABEL_77:
        LODWORD(v88[0]) = *((_DWORD *)v75 + 2);
        sub_C8CD80((__int64)&v88[1], (__int64)&v88[5], (__int64)(v75 + 2), (__int64)v28, 0, v29);
        sub_C8CD80((__int64)&v88[7], (__int64)&v88[11], (__int64)(v75 + 8), v77, v78, v79);
        v80 = BYTE1(v88[0]);
LABEL_78:
        v87 = v80;
        sub_26740A0((__int64)v88);
        return v87;
      }
      v83 = v74 - 1;
      v84 = 1;
      v85 = 0;
      while ( v76 != -4096 )
      {
        v86 = v84 + 1;
        v85 = v83 & (v84 + v85);
        v28 = &v75[16 * (unsigned __int64)v85];
        v76 = *v28;
        if ( !*v28 )
        {
          v75 += 16 * (unsigned __int64)v85;
          goto LABEL_77;
        }
        v84 = v86;
      }
    }
    memset(v88, 0, 0x78u);
    BYTE4(v88[4]) = 1;
    v88[2] = (__int64)&v88[5];
    v88[8] = (__int64)&v88[11];
    v80 = v17;
    LODWORD(v88[0]) = 65793;
    LODWORD(v88[3]) = 2;
    LODWORD(v88[9]) = 4;
    BYTE4(v88[10]) = 1;
    goto LABEL_78;
  }
  v33 = *(_QWORD *)(v31 + 16);
  if ( v33 )
  {
    while ( 1 )
    {
      v34 = *(_QWORD *)(v33 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v34 - 30) <= 0xAu )
        break;
      v33 = *(_QWORD *)(v33 + 8);
      if ( !v33 )
        return v17;
    }
LABEL_28:
    v35 = *(_DWORD *)(a1 + 248);
    v36 = *(_QWORD *)(v34 + 40);
    v37 = *(_QWORD *)(a1 + 232);
    if ( !v35 )
      goto LABEL_35;
    v38 = v35 - 1;
    v39 = (v35 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
    v40 = (__int64 *)(v37 + ((unsigned __int64)v39 << 7));
    v41 = *v40;
    if ( *v40 != v36 )
    {
      for ( i = 1; ; i = (unsigned int)v29 )
      {
        if ( v41 == -4096 )
          goto LABEL_35;
        v29 = (unsigned int)(i + 1);
        v39 = v38 & (i + v39);
        v40 = (__int64 *)(v37 + ((unsigned __int64)v39 << 7));
        v41 = *v40;
        if ( v36 == *v40 )
          break;
      }
    }
    LODWORD(v88[0]) = *((_DWORD *)v40 + 2);
    sub_C8CD80((__int64)&v88[1], (__int64)&v88[5], (__int64)(v40 + 2), v36, i, v29);
    sub_C8CD80((__int64)&v88[7], (__int64)&v88[11], (__int64)(v40 + 8), v42, v43, v44);
    v45 = BYTE1(v88[0]);
    if ( !BYTE4(v88[10]) )
      _libc_free(v88[8]);
    if ( !BYTE4(v88[4]) )
      _libc_free(v88[2]);
    if ( v45 )
    {
LABEL_35:
      while ( 1 )
      {
        v33 = *(_QWORD *)(v33 + 8);
        if ( !v33 )
          break;
        v34 = *(_QWORD *)(v33 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v34 - 30) <= 0xAu )
          goto LABEL_28;
      }
    }
    else
    {
      return 0;
    }
  }
  return v17;
}
