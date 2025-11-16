// Function: sub_3964770
// Address: 0x3964770
//
void __fastcall sub_3964770(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // r14
  _QWORD *v7; // r15
  _QWORD *v8; // rax
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  _QWORD *v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rsi
  _QWORD **v17; // rdx
  _QWORD **v18; // rdi
  _QWORD *v19; // rax
  __int64 v20; // rsi
  unsigned __int64 v21; // rcx
  __int64 v22; // rcx
  __int64 v23; // rdx
  unsigned int v24; // eax
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 *v27; // r12
  __int64 *v28; // rbx
  __int64 v29; // r13
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  int v32; // r8d
  int v33; // r9d
  __int64 v34; // rax
  unsigned __int64 v35; // rdi
  __int64 *v36; // rsi
  __int64 *v37; // r13
  __int64 *v38; // rax
  __int64 v39; // rdx
  __int64 *v40; // r15
  _BYTE *v41; // rsi
  int v42; // eax
  int v43; // r8d
  __int64 v44; // rdi
  int v45; // r9d
  unsigned int v46; // eax
  __int64 v47; // rsi
  char v48; // al
  __int64 *v49; // rdx
  unsigned int v50; // esi
  int v51; // eax
  int v52; // eax
  __int64 v53; // rax
  __int64 *v54; // rax
  _QWORD *v55; // rdx
  __int64 v56; // rax
  __int64 v57; // [rsp+8h] [rbp-1F8h]
  __int64 v60; // [rsp+30h] [rbp-1D0h] BYREF
  __int64 v61; // [rsp+38h] [rbp-1C8h] BYREF
  _BYTE *v62; // [rsp+40h] [rbp-1C0h] BYREF
  __int64 v63; // [rsp+48h] [rbp-1B8h]
  _BYTE v64[64]; // [rsp+50h] [rbp-1B0h] BYREF
  __int64 *v65; // [rsp+90h] [rbp-170h] BYREF
  __int64 v66; // [rsp+98h] [rbp-168h]
  _BYTE v67[128]; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v68; // [rsp+120h] [rbp-E0h] BYREF
  __int64 *v69; // [rsp+128h] [rbp-D8h]
  __int64 *v70; // [rsp+130h] [rbp-D0h]
  __int64 v71; // [rsp+138h] [rbp-C8h]
  int v72; // [rsp+140h] [rbp-C0h]
  _BYTE v73[184]; // [rsp+148h] [rbp-B8h] BYREF

  v4 = a1;
  v5 = *(_QWORD *)(a2 + 8);
  v62 = v64;
  v63 = 0x800000000LL;
  if ( !v5 )
  {
    v68 = 0;
    v69 = (__int64 *)v73;
    v70 = (__int64 *)v73;
    v71 = 16;
    v72 = 0;
    goto LABEL_40;
  }
  do
  {
    v13 = sub_3961CF0(a1, v5, 0);
    if ( !v13 )
      goto LABEL_10;
    v14 = *(_QWORD **)(a4 + 16);
    v8 = *(_QWORD **)(a4 + 8);
    if ( v14 == v8 )
    {
      v7 = &v8[*(unsigned int *)(a4 + 28)];
      if ( v8 == v7 )
      {
        v55 = *(_QWORD **)(a4 + 8);
      }
      else
      {
        do
        {
          if ( v13 == *v8 )
            break;
          ++v8;
        }
        while ( v7 != v8 );
        v55 = v7;
      }
    }
    else
    {
      v7 = &v14[*(unsigned int *)(a4 + 24)];
      v8 = sub_16CC9F0(a4, v13);
      if ( v13 == *v8 )
      {
        v15 = *(_QWORD *)(a4 + 16);
        if ( v15 == *(_QWORD *)(a4 + 8) )
          v16 = *(unsigned int *)(a4 + 28);
        else
          v16 = *(unsigned int *)(a4 + 24);
        v55 = (_QWORD *)(v15 + 8 * v16);
      }
      else
      {
        v11 = *(_QWORD *)(a4 + 16);
        if ( v11 != *(_QWORD *)(a4 + 8) )
        {
          v8 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(a4 + 24));
          goto LABEL_6;
        }
        v8 = (_QWORD *)(v11 + 8LL * *(unsigned int *)(a4 + 28));
        v55 = v8;
      }
    }
    while ( v55 != v8 && *v8 >= 0xFFFFFFFFFFFFFFFELL )
      ++v8;
LABEL_6:
    if ( v8 != v7 )
    {
      v12 = (unsigned int)v63;
      if ( (unsigned int)v63 >= HIDWORD(v63) )
      {
        sub_16CD150((__int64)&v62, v64, 0, 8, v9, v10);
        v12 = (unsigned int)v63;
      }
      *(_QWORD *)&v62[8 * v12] = v5;
      LODWORD(v63) = v63 + 1;
    }
LABEL_10:
    v5 = *(_QWORD *)(v5 + 8);
  }
  while ( v5 );
  v17 = (_QWORD **)v62;
  v18 = (_QWORD **)&v62[8 * (unsigned int)v63];
  if ( v18 != (_QWORD **)v62 )
  {
    do
    {
      v19 = *v17;
      if ( **v17 )
      {
        v20 = v19[1];
        v21 = v19[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v21 = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = *(_QWORD *)(v20 + 16) & 3LL | v21;
      }
      *v19 = a3;
      if ( a3 )
      {
        v22 = *(_QWORD *)(a3 + 8);
        v19[1] = v22;
        if ( v22 )
          *(_QWORD *)(v22 + 16) = (unsigned __int64)(v19 + 1) | *(_QWORD *)(v22 + 16) & 3LL;
        v19[2] = (a3 + 8) | v19[2] & 3LL;
        *(_QWORD *)(a3 + 8) = v19;
      }
      ++v17;
    }
    while ( v17 != v18 );
  }
  v68 = 0;
  v71 = 16;
  v72 = 0;
  v23 = *(_QWORD *)(a2 + 8);
  v69 = (__int64 *)v73;
  v70 = (__int64 *)v73;
  if ( !v23 )
  {
LABEL_40:
    if ( *(_BYTE *)(a2 + 16) <= 0x17u )
      goto LABEL_35;
    v61 = a2;
    if ( sub_1AE9990(a2, 0) )
    {
      v65 = (__int64 *)v67;
      v66 = 0x1000000000LL;
      sub_14EF3D0((__int64)&v65, &v61);
      v24 = v66;
      if ( (_DWORD)v66 )
      {
        v57 = v4;
        do
        {
          v25 = v65[v24 - 1];
          LODWORD(v66) = v24 - 1;
          v26 = 3LL * (*(_DWORD *)(v25 + 20) & 0xFFFFFFF);
          if ( (*(_BYTE *)(v25 + 23) & 0x40) != 0 )
          {
            v27 = *(__int64 **)(v25 - 8);
            v28 = &v27[v26];
          }
          else
          {
            v28 = (__int64 *)v25;
            v27 = (__int64 *)(v25 - v26 * 8);
          }
          for ( ; v28 != v27; LODWORD(v66) = v66 + 1 )
          {
            while ( 1 )
            {
              v29 = *v27;
              if ( *v27 )
              {
                v30 = v27[1];
                v31 = v27[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v31 = v30;
                if ( v30 )
                  *(_QWORD *)(v30 + 16) = *(_QWORD *)(v30 + 16) & 3LL | v31;
              }
              *v27 = 0;
              if ( !*(_QWORD *)(v29 + 8) && *(_BYTE *)(v29 + 16) > 0x17u && sub_1AE9990(v29, 0) )
                break;
              v27 += 3;
              if ( v28 == v27 )
                goto LABEL_58;
            }
            v34 = (unsigned int)v66;
            if ( (unsigned int)v66 >= HIDWORD(v66) )
            {
              sub_16CD150((__int64)&v65, v67, 0, 8, v32, v33);
              v34 = (unsigned int)v66;
            }
            v27 += 3;
            v65[v34] = v29;
          }
LABEL_58:
          sub_1412190((__int64)&v68, v25);
          sub_15F2070((_QWORD *)v25);
          v24 = v66;
        }
        while ( (_DWORD)v66 );
        v4 = v57;
      }
      if ( v65 != (__int64 *)v67 )
        _libc_free((unsigned __int64)v65);
      v35 = (unsigned __int64)v70;
      v36 = v69;
      if ( v70 == v69 )
        v37 = &v70[HIDWORD(v71)];
      else
        v37 = &v70[(unsigned int)v71];
      if ( v70 != v37 )
      {
        v38 = v70;
        while ( 1 )
        {
          v39 = *v38;
          v40 = v38;
          if ( (unsigned __int64)*v38 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v37 == ++v38 )
            goto LABEL_86;
        }
        if ( v38 != v37 )
        {
          while ( 1 )
          {
            v60 = v39;
            v41 = *(_BYTE **)(v4 + 304);
            if ( v41 == *(_BYTE **)(v4 + 312) )
            {
              sub_170B610(v4 + 296, v41, &v60);
              v39 = v60;
            }
            else
            {
              if ( v41 )
              {
                *(_QWORD *)v41 = v39;
                v41 = *(_BYTE **)(v4 + 304);
              }
              *(_QWORD *)(v4 + 304) = v41 + 8;
            }
            v61 = v39;
            if ( v39 )
            {
              v42 = *(_DWORD *)(v4 + 104);
              if ( v42 )
              {
                v43 = v42 - 1;
                v44 = *(_QWORD *)(v4 + 88);
                v45 = 1;
                v46 = (v42 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
                v47 = *(_QWORD *)(v44 + 16LL * v46);
                if ( v39 != v47 )
                {
                  while ( v47 != -8 )
                  {
                    v46 = v43 & (v45 + v46);
                    v47 = *(_QWORD *)(v44 + 16LL * v46);
                    if ( v47 == v39 )
                      goto LABEL_76;
                    ++v45;
                  }
                  goto LABEL_82;
                }
LABEL_76:
                v48 = sub_39632E0(v4 + 80, &v61, &v65);
                v49 = v65;
                if ( !v48 )
                {
                  v50 = *(_DWORD *)(v4 + 104);
                  v51 = *(_DWORD *)(v4 + 96);
                  ++*(_QWORD *)(v4 + 80);
                  v52 = v51 + 1;
                  if ( 4 * v52 >= 3 * v50 )
                  {
                    v50 *= 2;
                  }
                  else if ( v50 - *(_DWORD *)(v4 + 100) - v52 > v50 >> 3 )
                  {
LABEL_79:
                    *(_DWORD *)(v4 + 96) = v52;
                    if ( *v49 != -8 )
                      --*(_DWORD *)(v4 + 100);
                    v53 = v61;
                    v49[1] = 0;
                    *v49 = v53;
                    goto LABEL_82;
                  }
                  sub_39645B0(v4 + 80, v50);
                  sub_39632E0(v4 + 80, &v61, &v65);
                  v49 = v65;
                  v52 = *(_DWORD *)(v4 + 96) + 1;
                  goto LABEL_79;
                }
                v56 = v65[1];
                if ( v56 )
                  *(_BYTE *)(v56 + 41) = 1;
              }
            }
LABEL_82:
            v54 = v40 + 1;
            if ( v40 + 1 != v37 )
            {
              while ( 1 )
              {
                v39 = *v54;
                v40 = v54;
                if ( (unsigned __int64)*v54 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v37 == ++v54 )
                  goto LABEL_85;
              }
              if ( v37 != v54 )
                continue;
            }
            goto LABEL_85;
          }
        }
      }
    }
    else
    {
LABEL_85:
      v35 = (unsigned __int64)v70;
      v36 = v69;
    }
LABEL_86:
    if ( v36 != (__int64 *)v35 )
      _libc_free(v35);
  }
LABEL_35:
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
}
