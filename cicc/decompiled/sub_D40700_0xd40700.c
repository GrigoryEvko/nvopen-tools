// Function: sub_D40700
// Address: 0xd40700
//
__int64 __fastcall sub_D40700(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v7; // rsi
  _QWORD *v9; // rbx
  _BYTE *v10; // r13
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // rsi
  int v13; // r15d
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r9
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r11
  unsigned int v21; // edi
  __int64 *v22; // rcx
  __int64 v23; // r10
  __int64 v24; // r11
  unsigned __int64 *v25; // rdi
  unsigned __int64 *v26; // r8
  __int64 v27; // r9
  unsigned int v28; // r11d
  __int64 v29; // r10
  __int64 v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rdx
  unsigned int v35; // r8d
  __int64 v36; // rdx
  int v37; // ecx
  unsigned __int64 *v38; // rbx
  __int64 v39; // rax
  unsigned __int8 **v40; // r10
  unsigned __int8 **i; // r12
  __int64 v42; // r9
  __int64 v43; // rax
  int v44; // edx
  unsigned __int64 *v45; // rax
  unsigned __int8 *v46; // r8
  unsigned __int64 v47; // r11
  unsigned __int64 *v48; // rax
  unsigned __int8 **v50; // rbx
  char v51; // al
  __int64 v52; // rax
  _QWORD *v53; // r12
  _QWORD *v54; // r13
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // r12
  __int64 v58; // r13
  __int64 v59; // rdi
  unsigned __int64 v60; // [rsp+8h] [rbp-1B8h]
  _QWORD *v61; // [rsp+10h] [rbp-1B0h]
  unsigned int v62; // [rsp+18h] [rbp-1A8h]
  char v63; // [rsp+1Eh] [rbp-1A2h]
  unsigned __int8 v64; // [rsp+1Fh] [rbp-1A1h]
  __int64 v65; // [rsp+20h] [rbp-1A0h]
  __int64 v66; // [rsp+38h] [rbp-188h]
  unsigned __int64 *v67; // [rsp+50h] [rbp-170h]
  __int64 v68; // [rsp+58h] [rbp-168h]
  char v69; // [rsp+60h] [rbp-160h]
  unsigned __int8 *v70; // [rsp+60h] [rbp-160h]
  int v72; // [rsp+70h] [rbp-150h]
  __int64 v73; // [rsp+70h] [rbp-150h]
  unsigned __int64 v74; // [rsp+70h] [rbp-150h]
  int v76; // [rsp+80h] [rbp-140h]
  unsigned __int8 **v77; // [rsp+80h] [rbp-140h]
  int v78; // [rsp+88h] [rbp-138h]
  int v79; // [rsp+8Ch] [rbp-134h]
  unsigned int v80; // [rsp+9Ch] [rbp-124h] BYREF
  __int64 v81; // [rsp+A0h] [rbp-120h] BYREF
  __int64 v82; // [rsp+A8h] [rbp-118h]
  __int64 v83; // [rsp+B0h] [rbp-110h]
  unsigned int v84; // [rsp+B8h] [rbp-108h]
  unsigned __int64 *v85; // [rsp+C0h] [rbp-100h] BYREF
  __int64 v86; // [rsp+C8h] [rbp-F8h]
  _BYTE v87[32]; // [rsp+D0h] [rbp-F0h] BYREF
  _BYTE *v88; // [rsp+F0h] [rbp-D0h] BYREF
  int v89; // [rsp+F8h] [rbp-C8h]
  _BYTE v90[64]; // [rsp+100h] [rbp-C0h] BYREF
  unsigned __int8 **v91; // [rsp+140h] [rbp-80h] BYREF
  __int64 v92; // [rsp+148h] [rbp-78h]
  _BYTE v93[112]; // [rsp+150h] [rbp-70h] BYREF

  v5 = a1;
  v7 = a1 + 984;
  v61 = (_QWORD *)a5;
  v62 = *(_DWORD *)(a1 + 64);
  v66 = a1 + 984;
  v68 = *(_QWORD *)(a1 + 992);
  if ( v68 == a1 + 984 )
  {
    v28 = *(_DWORD *)(a2 + 16);
    v64 = 1;
    v63 = 0;
    if ( v28 )
      goto LABEL_29;
LABEL_43:
    v64 = 1;
    *(_BYTE *)a2 = *(_DWORD *)(a2 + 304) != 0;
    return v64;
  }
  v63 = 0;
  v64 = 1;
  v78 = 0;
  v65 = a2;
  do
  {
    v79 = v78 + 1;
    sub_FDA300(&v88, v68);
    v80 = 1;
    v81 = 0;
    v91 = (unsigned __int8 **)v93;
    v92 = 0x400000000LL;
    v85 = (unsigned __int64 *)v87;
    v9 = v88;
    v86 = 0x400000000LL;
    v82 = 0;
    v83 = 0;
    v10 = &v88[8 * v89];
    v84 = 0;
    if ( v88 == v10 )
    {
LABEL_24:
      v7 = 16LL * v84;
      sub_C7D6A0(v82, v7, 8);
      if ( v88 != v90 )
        _libc_free(v88, v7);
      ++v78;
      goto LABEL_27;
    }
    v76 = 0;
    v11 = 4;
    v12 = 0;
    v13 = 0;
    while ( 1 )
    {
      v17 = *(unsigned int *)(a1 + 24);
      v18 = *(_QWORD *)(a1 + 8);
      v19 = *v9 | 4LL;
      v20 = *v9 & 0xFFFFFFFFFFFFFFFBLL;
      if ( !(_DWORD)v17 )
        break;
      v21 = (v17 - 1) & (v19 ^ (v19 >> 9));
      v22 = (__int64 *)(v18 + 16LL * v21);
      v23 = *v22;
      if ( v19 != *v22 )
      {
        v37 = 1;
        while ( v23 != -4 )
        {
          v21 = (v17 - 1) & (v37 + v21);
          v72 = v37 + 1;
          v22 = (__int64 *)(v18 + 16LL * v21);
          v23 = *v22;
          if ( v19 == *v22 )
            goto LABEL_12;
          v37 = v72;
        }
        break;
      }
LABEL_12:
      if ( v22 == (__int64 *)(v18 + 16 * v17) )
        break;
      v14 = (unsigned int)v12;
      ++v13;
      v15 = 1;
      if ( (unsigned int)v12 < v11 )
        goto LABEL_6;
LABEL_14:
      v24 = (4 * v15) | v20;
      if ( v11 < v14 + 1 )
      {
        v12 = (unsigned __int64)v87;
        v73 = v24;
        sub_C8D5F0((__int64)&v85, v87, v14 + 1, 8u, v11, v14 + 1);
        v14 = (unsigned int)v86;
        v24 = v73;
      }
      ++v9;
      v85[v14] = v24;
      LODWORD(v86) = v86 + 1;
      if ( v10 == (_BYTE *)v9 )
        goto LABEL_17;
LABEL_9:
      v12 = (unsigned int)v86;
      v11 = HIDWORD(v86);
    }
    v14 = (unsigned int)v12;
    ++v76;
    v15 = 0;
    if ( (unsigned int)v12 >= v11 )
      goto LABEL_14;
LABEL_6:
    v16 = &v85[v14];
    if ( v16 )
    {
      *v16 = (4 * v15) | v20;
      LODWORD(v12) = v86;
    }
    v12 = (unsigned int)(v12 + 1);
    ++v9;
    LODWORD(v86) = v12;
    if ( v10 != (_BYTE *)v9 )
      goto LABEL_9;
LABEL_17:
    v25 = v85;
    v26 = v85;
    if ( !v13 || v13 == 1 && !v76 )
    {
      if ( v85 != (unsigned __int64 *)v87 )
        _libc_free(v85, v12);
      if ( v91 != (unsigned __int8 **)v93 )
        _libc_free(v91, v12);
      goto LABEL_24;
    }
    v67 = &v85[(unsigned int)v86];
    if ( v67 == v85 )
    {
      if ( v80 <= 2 )
      {
        if ( (_DWORD)v92 )
        {
LABEL_74:
          v63 = 1;
          v26 = v25;
        }
        else
        {
          v69 = 1;
LABEL_62:
          v64 &= v69;
          v26 = v25;
        }
      }
      else
      {
        v63 = 1;
      }
      goto LABEL_63;
    }
    v69 = 1;
    v38 = v85;
    while ( 2 )
    {
      v12 = (unsigned __int64)v38;
      v39 = sub_D40250(a1, v38);
      v40 = *(unsigned __int8 ***)(v39 + 32);
      v77 = &v40[*(unsigned int *)(v39 + 40)];
      if ( v40 != v77 )
      {
        for ( i = *(unsigned __int8 ***)(v39 + 32); v77 != i; ++i )
        {
          while ( 1 )
          {
            v12 = v65;
            if ( !(unsigned __int8)sub_D3E300(a1, v65, *v38, *i, a4, (__int64)&v81, a3, &v80, v79, 0) )
              break;
LABEL_52:
            if ( v77 == ++i )
              goto LABEL_58;
          }
          v43 = (unsigned int)v92;
          v44 = v92;
          if ( (unsigned int)v92 >= (unsigned __int64)HIDWORD(v92) )
          {
            v46 = *i;
            v47 = *v38;
            if ( HIDWORD(v92) < (unsigned __int64)(unsigned int)v92 + 1 )
            {
              v12 = (unsigned __int64)v93;
              v60 = *v38;
              v70 = *i;
              sub_C8D5F0((__int64)&v91, v93, (unsigned int)v92 + 1LL, 0x10u, (__int64)v46, v42);
              v43 = (unsigned int)v92;
              v47 = v60;
              v46 = v70;
            }
            v48 = (unsigned __int64 *)&v91[2 * v43];
            v69 = 0;
            *v48 = v47;
            v48[1] = (unsigned __int64)v46;
            LODWORD(v92) = v92 + 1;
            goto LABEL_52;
          }
          v45 = (unsigned __int64 *)&v91[2 * (unsigned int)v92];
          if ( v45 )
          {
            *v45 = *v38;
            v45[1] = (unsigned __int64)*i;
            v44 = v92;
          }
          v69 = 0;
          LODWORD(v92) = v44 + 1;
        }
      }
LABEL_58:
      if ( v67 != ++v38 )
        continue;
      break;
    }
    if ( v80 <= 2 )
    {
      v12 = (unsigned int)v92;
      if ( !(_DWORD)v92 )
      {
        v25 = v85;
        goto LABEL_62;
      }
    }
    if ( v69 )
    {
      v25 = v85;
      goto LABEL_74;
    }
    v12 = (unsigned __int64)&v91[2 * (unsigned int)v92];
    v74 = v12;
    if ( v91 == (unsigned __int8 **)v12 )
    {
      v63 = 1;
      v26 = v85;
    }
    else
    {
      v50 = v91;
      do
      {
        v12 = v65;
        v51 = sub_D3E300(a1, v65, (__int64)*v50, v50[1], a4, (__int64)&v81, a3, &v80, v79, 1);
        if ( !v51 )
        {
          v64 = 0;
          v63 = 1;
          v26 = v85;
          *v61 = (unsigned __int64)*v50 & 0xFFFFFFFFFFFFFFF8LL;
          goto LABEL_63;
        }
        v50 += 2;
      }
      while ( (unsigned __int8 **)v74 != v50 );
      v63 = v51;
      v26 = v85;
    }
LABEL_63:
    v78 += 2;
    if ( v26 != (unsigned __int64 *)v87 )
      _libc_free(v26, v12);
    if ( v91 != (unsigned __int8 **)v93 )
      _libc_free(v91, v12);
    v7 = 16LL * v84;
    sub_C7D6A0(v82, v7, 8);
    if ( v88 != v90 )
      _libc_free(v88, v7);
LABEL_27:
    v68 = *(_QWORD *)(v68 + 8);
  }
  while ( v66 != v68 );
  a2 = v65;
  v5 = a1;
  v28 = *(_DWORD *)(v65 + 16);
  if ( v28 )
  {
LABEL_29:
    v27 = 1;
    v29 = 0;
    if ( v28 > 1 )
    {
      while ( 2 )
      {
        v30 = *(_QWORD *)(a2 + 8);
        v31 = v30 + v29;
        v29 += 72;
        v7 = *(unsigned int *)(v31 + 44);
        v32 = v30 + v29;
        v33 = v30 + 72 * (v27 + v28 - 1 - (unsigned int)v27) + 72;
        do
        {
          if ( (_DWORD)v7 != *(_DWORD *)(v32 + 44) && *(_DWORD *)(v31 + 48) == *(_DWORD *)(v32 + 48) )
          {
            v34 = *(_QWORD *)(*(_QWORD *)(v31 + 16) + 8LL);
            if ( (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17 <= 1 )
              v34 = **(_QWORD **)(v34 + 16);
            v35 = *(_DWORD *)(v34 + 8);
            v36 = *(_QWORD *)(*(_QWORD *)(v32 + 16) + 8LL);
            a5 = v35 >> 8;
            if ( (unsigned int)*(unsigned __int8 *)(v36 + 8) - 17 <= 1 )
              v36 = **(_QWORD **)(v36 + 16);
            if ( (_DWORD)a5 != *(_DWORD *)(v36 + 8) >> 8 )
              return 0;
          }
          v32 += 72;
        }
        while ( v33 != v32 );
        if ( v28 > (unsigned int)++v27 )
          continue;
        break;
      }
    }
  }
  if ( !v63 )
  {
    if ( !v64 )
    {
      *(_BYTE *)a2 = 0;
      return 1;
    }
    goto LABEL_43;
  }
  if ( v64 )
  {
    sub_D3F9D0(a2, *(_QWORD *)(v5 + 1056), v62 != 0, v62, a5, v27);
    goto LABEL_43;
  }
  v52 = *(unsigned int *)(a2 + 16);
  v53 = *(_QWORD **)(a2 + 8);
  *(_BYTE *)a2 = 0;
  *(_BYTE *)(a2 + 376) = 1;
  v54 = &v53[9 * v52];
  while ( v53 != v54 )
  {
    while ( 1 )
    {
      v55 = *(v54 - 7);
      v54 -= 9;
      if ( v55 == -4096 || v55 == 0 || v55 == -8192 )
        break;
      sub_BD60C0(v54);
      if ( v53 == v54 )
        goto LABEL_94;
    }
  }
LABEL_94:
  v56 = *(unsigned int *)(a2 + 176);
  v57 = *(_QWORD *)(a2 + 168);
  *(_DWORD *)(a2 + 16) = 0;
  *(_DWORD *)(a2 + 304) = 0;
  *(_DWORD *)(a2 + 392) = 0;
  v56 *= 48;
  v58 = v57 + v56;
  while ( v57 != v58 )
  {
    v58 -= 48;
    v59 = *(_QWORD *)(v58 + 16);
    if ( v59 != v58 + 32 )
      _libc_free(v59, v7);
  }
  *(_DWORD *)(a2 + 176) = 0;
  return v64;
}
