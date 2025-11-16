// Function: sub_2BAFCA0
// Address: 0x2bafca0
//
void __fastcall sub_2BAFCA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // r12
  bool v7; // zf
  unsigned __int64 v8; // rsi
  _BYTE *v9; // rax
  _BYTE **v10; // rbx
  __int64 v11; // r8
  _BYTE **v12; // r15
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned int *v15; // r13
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // r15
  __int64 v20; // rbx
  __int64 v21; // rsi
  _QWORD *v22; // rax
  __int64 v23; // rax
  char *v24; // rbx
  char *v25; // r15
  char *v26; // r12
  __int64 v27; // rdx
  __int64 v28; // rax
  int v29; // ecx
  __int64 v30; // rdx
  __int64 v31; // r10
  int v32; // r11d
  char **v33; // r15
  char **v34; // r12
  char *v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  _BYTE *v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // rbx
  _BYTE *v41; // r8
  unsigned __int64 *v42; // r12
  unsigned int *v43; // rbx
  __int64 v44; // r8
  unsigned int v45; // esi
  unsigned __int64 v46; // rdx
  __int64 v47; // r15
  unsigned __int64 v48; // r8
  void *v49; // r11
  size_t v50; // r10
  char *v51; // rdi
  _BYTE *v52; // rbx
  unsigned __int64 v53; // r12
  size_t n; // [rsp+10h] [rbp-1D0h]
  char *src; // [rsp+18h] [rbp-1C8h]
  void *srca; // [rsp+18h] [rbp-1C8h]
  _BYTE **v57; // [rsp+20h] [rbp-1C0h]
  char *v58; // [rsp+20h] [rbp-1C0h]
  char *v59; // [rsp+20h] [rbp-1C0h]
  unsigned int *v60; // [rsp+28h] [rbp-1B8h]
  unsigned __int64 v62; // [rsp+30h] [rbp-1B0h]
  unsigned __int64 v63; // [rsp+30h] [rbp-1B0h]
  unsigned int *v64; // [rsp+38h] [rbp-1A8h]
  char v65; // [rsp+4Fh] [rbp-191h] BYREF
  __int64 v66; // [rsp+50h] [rbp-190h] BYREF
  __int64 v67; // [rsp+58h] [rbp-188h] BYREF
  __int64 v68[6]; // [rsp+60h] [rbp-180h] BYREF
  char **v69; // [rsp+90h] [rbp-150h] BYREF
  unsigned int v70; // [rsp+98h] [rbp-148h]
  char v71; // [rsp+A0h] [rbp-140h] BYREF
  char *v72; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v73; // [rsp+D8h] [rbp-108h]
  _BYTE v74[48]; // [rsp+E0h] [rbp-100h] BYREF
  _BYTE *v75; // [rsp+110h] [rbp-D0h] BYREF
  __int64 v76; // [rsp+118h] [rbp-C8h]
  _BYTE v77[64]; // [rsp+120h] [rbp-C0h] BYREF
  _BYTE *v78; // [rsp+160h] [rbp-80h] BYREF
  __int64 v79; // [rsp+168h] [rbp-78h]
  _BYTE v80[112]; // [rsp+170h] [rbp-70h] BYREF

  v6 = a1;
  v7 = *(_BYTE *)(a1 + 1256) == 0;
  *(_DWORD *)(a1 + 1252) = *(_DWORD *)(a1 + 8);
  if ( v7 )
    *(_BYTE *)(a1 + 1256) = 1;
  v8 = *(unsigned int *)(a1 + 1240);
  v75 = v77;
  v76 = 0x100000000LL;
  if ( v8 )
  {
    v9 = v77;
    v10 = &v78;
    v11 = v8 << 6;
    if ( v8 == 1
      || (sub_2B542E0((__int64)&v75, v8, a3, a4, v11, a6),
          v12 = (_BYTE **)v75,
          v11 = v8 << 6,
          v9 = &v75[64 * (unsigned __int64)(unsigned int)v76],
          v10 = (_BYTE **)&v75[64 * v8],
          v9 != (_BYTE *)v10) )
    {
      do
      {
        if ( v9 )
        {
          *(_QWORD *)v9 = 0;
          *((_QWORD *)v9 + 1) = v9 + 32;
          *((_DWORD *)v9 + 4) = 4;
          *((_DWORD *)v9 + 5) = 0;
          *((_DWORD *)v9 + 6) = 0;
          v9[28] = 1;
        }
        v9 += 64;
      }
      while ( v9 != (_BYTE *)v10 );
      v12 = (_BYTE **)v75;
      v10 = (_BYTE **)&v75[v11];
    }
    v13 = *(_QWORD *)(a1 + 1232);
    v14 = *(unsigned int *)(a1 + 1240);
    LODWORD(v76) = v8;
    v60 = (unsigned int *)(v13 + 4 * v14);
    if ( (unsigned int *)v13 != v60 && v12 != v10 )
    {
      v57 = v10;
      v15 = (unsigned int *)v13;
      v16 = (__int64)v12;
      while ( 1 )
      {
        v17 = *v15;
        v18 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v17);
        v19 = *(__int64 **)v18;
        v20 = *(_QWORD *)v18 + 8LL * *(unsigned int *)(v18 + 8);
        if ( *(_QWORD *)v18 != v20 )
          break;
LABEL_20:
        ++v15;
        v16 += 64;
        if ( (_BYTE **)v16 == v57 || v15 == v60 )
          goto LABEL_22;
      }
      a6 = *(unsigned __int8 *)(v16 + 28);
      while ( 1 )
      {
        while ( 1 )
        {
          v21 = *v19;
          if ( (_BYTE)a6 )
            break;
LABEL_75:
          ++v19;
          sub_C8CC70(v16, v21, v17, v13, v11, a6);
          a6 = *(unsigned __int8 *)(v16 + 28);
          if ( (__int64 *)v20 == v19 )
            goto LABEL_20;
        }
        v22 = *(_QWORD **)(v16 + 8);
        v13 = *(unsigned int *)(v16 + 20);
        v17 = (__int64)&v22[v13];
        if ( v22 == (_QWORD *)v17 )
        {
LABEL_77:
          if ( (unsigned int)v13 >= *(_DWORD *)(v16 + 16) )
            goto LABEL_75;
          v13 = (unsigned int)(v13 + 1);
          ++v19;
          *(_DWORD *)(v16 + 20) = v13;
          *(_QWORD *)v17 = v21;
          a6 = *(unsigned __int8 *)(v16 + 28);
          ++*(_QWORD *)v16;
          if ( (__int64 *)v20 == v19 )
            goto LABEL_20;
        }
        else
        {
          while ( v21 != *v22 )
          {
            if ( (_QWORD *)v17 == ++v22 )
              goto LABEL_77;
          }
          if ( (__int64 *)v20 == ++v19 )
            goto LABEL_20;
        }
      }
    }
  }
LABEL_22:
  v23 = *(_QWORD *)(a1 + 3296);
  v67 = a1;
  v68[3] = a1;
  v66 = v23;
  v68[0] = v23;
  v68[1] = (__int64)&v65;
  v68[2] = (__int64)&v66;
  v68[4] = (__int64)&v67;
  v68[5] = (__int64)&v75;
  v24 = *(char **)(a2 + 272);
  v25 = &v24[104 * *(unsigned int *)(a2 + 280)];
  if ( v25 != v24 )
  {
    v26 = *(char **)(a2 + 272);
    do
    {
      sub_2BACC10((__int64)&v69, v68, *((_QWORD *)v26 + 3), *((unsigned int *)v26 + 8), 0, a6);
      v27 = *((unsigned int *)v26 + 8);
      if ( (_DWORD)v27 )
      {
        if ( v70 )
        {
          v28 = *((_QWORD *)v26 + 3);
          v29 = 0;
          v30 = v28 + (v27 << 6);
          do
          {
            v29 += *(_DWORD *)(v28 + 8);
            v28 += 64;
          }
          while ( v28 != v30 );
          if ( v70 != v29 )
          {
            sub_2B08900((__int64)v69, v70);
            sub_2B08680(*(_QWORD *)(*(_QWORD *)v31 + 8LL), v32);
            if ( (unsigned __int8)sub_DFA510(v66) )
            {
              if ( !(unsigned __int8)sub_DFA5A0(v66) )
              {
                v78 = v80;
                v79 = 0x100000000LL;
                if ( &v69[v70] == v69 )
                {
                  v38 = v80;
                  v39 = 0;
                }
                else
                {
                  v58 = v25;
                  v33 = &v69[v70];
                  src = v26;
                  v34 = v69;
                  do
                  {
                    v35 = *v34;
                    v36 = *(_QWORD *)(a1 + 3344);
                    ++v34;
                    v37 = *(_QWORD *)(a1 + 3288);
                    v72 = v35;
                    sub_2B5E1B0(a1, &v72, 1, v36, v37, (__int64)&v78, 0);
                  }
                  while ( v33 != v34 );
                  v25 = v58;
                  v38 = v78;
                  v26 = src;
                  v39 = (unsigned int)v79;
                }
                sub_2BACC10((__int64)&v72, v68, (__int64)v38, v39, 1u, a6);
                if ( v72 != v74 )
                  _libc_free((unsigned __int64)v72);
                v40 = (__int64)v78;
                v41 = &v78[64 * (unsigned __int64)(unsigned int)v79];
                if ( v78 != v41 )
                {
                  v59 = v26;
                  v42 = (unsigned __int64 *)&v78[64 * (unsigned __int64)(unsigned int)v79];
                  do
                  {
                    v42 -= 8;
                    if ( (unsigned __int64 *)*v42 != v42 + 2 )
                      _libc_free(*v42);
                  }
                  while ( (unsigned __int64 *)v40 != v42 );
                  v26 = v59;
                  v41 = v78;
                }
                if ( v41 != v80 )
                  _libc_free((unsigned __int64)v41);
              }
            }
          }
        }
      }
      if ( v69 != (char **)&v71 )
        _libc_free((unsigned __int64)v69);
      v26 += 104;
    }
    while ( v25 != v26 );
    v6 = a1;
  }
  v43 = *(unsigned int **)(v6 + 1232);
  v64 = &v43[*(unsigned int *)(v6 + 1240)];
  if ( v64 != v43 )
  {
    while ( 1 )
    {
      v47 = *(_QWORD *)(*(_QWORD *)v6 + 8LL * *v43);
      v48 = *(unsigned int *)(v47 + 8);
      v49 = *(void **)v47;
      v72 = v74;
      v73 = 0x600000000LL;
      v50 = 8 * v48;
      if ( v48 > 6 )
        break;
      if ( v50 )
      {
        v51 = v74;
        goto LABEL_58;
      }
LABEL_52:
      v44 = v50 + v48;
      LODWORD(v73) = v44;
      v45 = *(_DWORD *)(v47 + 152);
      v46 = (unsigned int)v44;
      if ( v45 )
      {
        v79 = 0xC00000000LL;
        v78 = v80;
        sub_2B0FC00(*(_QWORD *)(v47 + 144), v45, (__int64)&v78, 0xC00000000LL, v44, a6);
        sub_2B38DA0((unsigned int *)&v72, (__int64)v78);
        if ( v78 != v80 )
          _libc_free((unsigned __int64)v78);
        v46 = (unsigned int)v73;
      }
      v78 = 0;
      LODWORD(v79) = -1;
      sub_2BA65A0(v6, (__int64 *)v72, v46, 0, &v78, 0);
      if ( v72 != v74 )
        _libc_free((unsigned __int64)v72);
      if ( v64 == ++v43 )
        goto LABEL_63;
    }
    n = 8 * v48;
    srca = v49;
    v62 = v48;
    sub_C8D5F0((__int64)&v72, v74, v48, 8u, v48, a6);
    v48 = v62;
    v49 = srca;
    v50 = n;
    v51 = &v72[8 * (unsigned int)v73];
LABEL_58:
    v63 = v48;
    memcpy(v51, v49, v50);
    v50 = (unsigned int)v73;
    v48 = v63;
    goto LABEL_52;
  }
LABEL_63:
  if ( *(_DWORD *)(v6 + 1252) == *(_DWORD *)(v6 + 8) && *(_BYTE *)(v6 + 1256) )
    *(_BYTE *)(v6 + 1256) = 0;
  v52 = v75;
  v53 = (unsigned __int64)&v75[64 * (unsigned __int64)(unsigned int)v76];
  if ( v75 != (_BYTE *)v53 )
  {
    do
    {
      while ( 1 )
      {
        v53 -= 64LL;
        if ( !*(_BYTE *)(v53 + 28) )
          break;
        if ( v52 == (_BYTE *)v53 )
          goto LABEL_69;
      }
      _libc_free(*(_QWORD *)(v53 + 8));
    }
    while ( v52 != (_BYTE *)v53 );
LABEL_69:
    v53 = (unsigned __int64)v75;
  }
  if ( (_BYTE *)v53 != v77 )
    _libc_free(v53);
}
