// Function: sub_35EC3C0
// Address: 0x35ec3c0
//
_BOOL8 __fastcall sub_35EC3C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *i; // rdx
  int v10; // eax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *k; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  void (__fastcall *v17)(__int64 *, __int64, __int64); // rax
  __int64 v18; // r13
  unsigned int *v19; // rcx
  int v20; // edx
  _BYTE *v21; // rdx
  _BYTE *v22; // r12
  _BYTE *v23; // r15
  _BYTE *v24; // rbx
  _BYTE *v25; // r15
  unsigned int v27; // edx
  unsigned int *v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  int v31; // r12d
  unsigned int n; // ebx
  unsigned int v33; // edx
  _DWORD *v34; // rax
  unsigned __int64 v35; // rax
  int *v36; // rdi
  __int64 v37; // rsi
  unsigned __int64 v38; // rax
  int *v39; // rdi
  __int64 v40; // rsi
  unsigned int v41; // ecx
  unsigned int v42; // eax
  _QWORD *v43; // rdi
  int v44; // ebx
  _QWORD *v45; // rax
  unsigned int v46; // ecx
  unsigned int v47; // eax
  _QWORD *v48; // rdi
  int v49; // ebx
  unsigned __int64 v50; // rdx
  unsigned __int64 v51; // rax
  _QWORD *v52; // rax
  __int64 v53; // rdx
  _QWORD *j; // rdx
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // rax
  _QWORD *v57; // rax
  __int64 v58; // rdx
  _QWORD *m; // rdx
  _QWORD *v60; // rax
  __int64 v61; // [rsp+30h] [rbp-130h]
  bool v62; // [rsp+3Fh] [rbp-121h]
  unsigned int v63; // [rsp+44h] [rbp-11Ch] BYREF
  __int64 v64; // [rsp+48h] [rbp-118h] BYREF
  _BYTE v65[32]; // [rsp+50h] [rbp-110h] BYREF
  _BYTE *v66; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v67; // [rsp+78h] [rbp-E8h]
  _BYTE v68[40]; // [rsp+80h] [rbp-E0h] BYREF
  int v69; // [rsp+A8h] [rbp-B8h] BYREF
  unsigned __int64 v70; // [rsp+B0h] [rbp-B0h]
  int *v71; // [rsp+B8h] [rbp-A8h]
  int *v72; // [rsp+C0h] [rbp-A0h]
  __int64 v73; // [rsp+C8h] [rbp-98h]
  unsigned int *v74; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v75; // [rsp+D8h] [rbp-88h]
  _BYTE v76[40]; // [rsp+E0h] [rbp-80h] BYREF
  int v77; // [rsp+108h] [rbp-58h] BYREF
  unsigned __int64 v78; // [rsp+110h] [rbp-50h]
  int *v79; // [rsp+118h] [rbp-48h]
  int *v80; // [rsp+120h] [rbp-40h]
  __int64 v81; // [rsp+128h] [rbp-38h]

  v6 = *(_DWORD *)(a1 + 224);
  ++*(_QWORD *)(a1 + 208);
  *(_DWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 228) )
      goto LABEL_7;
    v7 = *(unsigned int *)(a1 + 232);
    if ( (unsigned int)v7 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 216), 16LL * *(unsigned int *)(a1 + 232), 8);
      *(_QWORD *)(a1 + 216) = 0;
      *(_QWORD *)(a1 + 224) = 0;
      *(_DWORD *)(a1 + 232) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v46 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 232);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v46 = 64;
  if ( (unsigned int)v7 <= v46 )
  {
LABEL_4:
    v8 = *(_QWORD **)(a1 + 216);
    for ( i = &v8[2 * v7]; i != v8; v8 += 2 )
      *v8 = -4096;
    *(_QWORD *)(a1 + 224) = 0;
    goto LABEL_7;
  }
  v47 = v6 - 1;
  if ( !v47 )
  {
    v48 = *(_QWORD **)(a1 + 216);
    v49 = 64;
LABEL_102:
    sub_C7D6A0((__int64)v48, 16 * v7, 8);
    v50 = ((((((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
             | (4 * v49 / 3u + 1)
             | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
           | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
           | (4 * v49 / 3u + 1)
           | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
           | (4 * v49 / 3u + 1)
           | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
         | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
         | (4 * v49 / 3u + 1)
         | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 16;
    v51 = (v50
         | (((((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
             | (4 * v49 / 3u + 1)
             | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
           | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
           | (4 * v49 / 3u + 1)
           | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
           | (4 * v49 / 3u + 1)
           | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 4)
         | (((4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1)) >> 2)
         | (4 * v49 / 3u + 1)
         | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 232) = v51;
    v52 = (_QWORD *)sub_C7D670(16 * v51, 8);
    v53 = *(unsigned int *)(a1 + 232);
    *(_QWORD *)(a1 + 224) = 0;
    *(_QWORD *)(a1 + 216) = v52;
    for ( j = &v52[2 * v53]; j != v52; v52 += 2 )
    {
      if ( v52 )
        *v52 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v47, v47);
  v48 = *(_QWORD **)(a1 + 216);
  v49 = 1 << (33 - (v47 ^ 0x1F));
  if ( v49 < 64 )
    v49 = 64;
  if ( v49 != (_DWORD)v7 )
    goto LABEL_102;
  *(_QWORD *)(a1 + 224) = 0;
  v60 = &v48[2 * (unsigned int)v49];
  do
  {
    if ( v48 )
      *v48 = -4096;
    v48 += 2;
  }
  while ( v60 != v48 );
LABEL_7:
  v10 = *(_DWORD *)(a1 + 256);
  ++*(_QWORD *)(a1 + 240);
  if ( !v10 )
  {
    if ( !*(_DWORD *)(a1 + 260) )
      goto LABEL_13;
    v11 = *(unsigned int *)(a1 + 264);
    if ( (unsigned int)v11 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 248), 16LL * *(unsigned int *)(a1 + 264), 8);
      *(_QWORD *)(a1 + 248) = 0;
      *(_QWORD *)(a1 + 256) = 0;
      *(_DWORD *)(a1 + 264) = 0;
      goto LABEL_13;
    }
    goto LABEL_10;
  }
  v41 = 4 * v10;
  v11 = *(unsigned int *)(a1 + 264);
  if ( (unsigned int)(4 * v10) < 0x40 )
    v41 = 64;
  if ( (unsigned int)v11 <= v41 )
  {
LABEL_10:
    v12 = *(_QWORD **)(a1 + 248);
    for ( k = &v12[2 * v11]; k != v12; v12 += 2 )
      *v12 = -4096;
    *(_QWORD *)(a1 + 256) = 0;
    goto LABEL_13;
  }
  v42 = v10 - 1;
  if ( v42 )
  {
    _BitScanReverse(&v42, v42);
    v43 = *(_QWORD **)(a1 + 248);
    v44 = 1 << (33 - (v42 ^ 0x1F));
    if ( v44 < 64 )
      v44 = 64;
    if ( (_DWORD)v11 == v44 )
    {
      *(_QWORD *)(a1 + 256) = 0;
      v45 = &v43[2 * (unsigned int)v11];
      do
      {
        if ( v43 )
          *v43 = -4096;
        v43 += 2;
      }
      while ( v45 != v43 );
      goto LABEL_13;
    }
  }
  else
  {
    v43 = *(_QWORD **)(a1 + 248);
    v44 = 64;
  }
  sub_C7D6A0((__int64)v43, 16LL * *(unsigned int *)(a1 + 264), 8);
  v55 = ((((((((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
           | (4 * v44 / 3u + 1)
           | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 4)
         | (((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
         | (4 * v44 / 3u + 1)
         | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
         | (4 * v44 / 3u + 1)
         | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 4)
       | (((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
       | (4 * v44 / 3u + 1)
       | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 16;
  v56 = (v55
       | (((((((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
           | (4 * v44 / 3u + 1)
           | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 4)
         | (((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
         | (4 * v44 / 3u + 1)
         | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
         | (4 * v44 / 3u + 1)
         | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 4)
       | (((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
       | (4 * v44 / 3u + 1)
       | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 264) = v56;
  v57 = (_QWORD *)sub_C7D670(16 * v56, 8);
  v58 = *(unsigned int *)(a1 + 264);
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 248) = v57;
  for ( m = &v57[2 * v58]; m != v57; v57 += 2 )
  {
    if ( v57 )
      *v57 = -4096;
  }
LABEL_13:
  *(_DWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 6440) = 0xFFFFFFFFLL;
  v14 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 6432) = 0;
  *(_DWORD *)(a1 + 6448) = 0;
  if ( *(_QWORD *)(v14 + 48) )
  {
    v15 = *(_QWORD *)(a1 + 24);
    v69 = 0;
    v71 = &v69;
    v72 = &v69;
    v66 = v68;
    v74 = (unsigned int *)v76;
    v16 = *(_QWORD *)(a1 + 48);
    v67 = 0x800000000LL;
    v75 = 0x800000000LL;
    v70 = 0;
    v73 = 0;
    v77 = 0;
    v78 = 0;
    v79 = &v77;
    v80 = &v77;
    v81 = 0;
    v17 = *(void (__fastcall **)(__int64 *, __int64, __int64))(*(_QWORD *)v16 + 376LL);
    if ( (char *)v17 == (char *)sub_2FDC520 )
    {
      v64 = 0;
    }
    else
    {
      v17(&v64, v16, v15);
      v15 = *(_QWORD *)(a1 + 24);
    }
    v18 = *(_QWORD *)(v15 + 56);
    v19 = &v63;
    v61 = v15 + 48;
    while ( v61 != v18 )
    {
      if ( (*(_QWORD *)(*(_QWORD *)(v18 + 16) + 24LL) & 0x10) == 0 )
      {
        v20 = *(_DWORD *)(v18 + 44);
        if ( (v20 & 4) != 0 || (v20 & 8) == 0 )
          v62 = (*(_QWORD *)(*(_QWORD *)(v18 + 16) + 24LL) & 0x200LL) != 0;
        else
          v62 = sub_2E88A90(v18, 512, 1);
        if ( !v62 )
        {
          if ( *(_WORD *)(v18 + 68) == 68 || !*(_WORD *)(v18 + 68) )
          {
            v27 = *(_DWORD *)(*(_QWORD *)(v18 + 32) + 8LL);
            if ( v81 )
            {
              v35 = v78;
              if ( v78 )
              {
                v36 = &v77;
                do
                {
                  while ( 1 )
                  {
                    v37 = *(_QWORD *)(v35 + 16);
                    v19 = *(unsigned int **)(v35 + 24);
                    if ( v27 <= *(_DWORD *)(v35 + 32) )
                      break;
                    v35 = *(_QWORD *)(v35 + 24);
                    if ( !v19 )
                      goto LABEL_72;
                  }
                  v36 = (int *)v35;
                  v35 = *(_QWORD *)(v35 + 16);
                }
                while ( v37 );
LABEL_72:
                if ( v36 != &v77 && v27 >= v36[8] )
                  goto LABEL_40;
              }
            }
            else
            {
              v28 = v74;
              v19 = &v74[(unsigned int)v75];
              if ( v74 != v19 )
              {
                while ( v27 != *v28 )
                {
                  if ( v19 == ++v28 )
                    goto LABEL_57;
                }
                if ( v28 != v19 )
                  goto LABEL_40;
              }
            }
LABEL_57:
            v63 = *(_DWORD *)(*(_QWORD *)(v18 + 32) + 8LL);
            sub_3361470((__int64)v65, (__int64)&v66, &v63, (__int64)v19, a5);
            v31 = *(_DWORD *)(v18 + 40) & 0xFFFFFF;
            if ( v31 != 1 )
            {
              for ( n = 1; v31 != n; n += 2 )
              {
                v33 = *(_DWORD *)(*(_QWORD *)(v18 + 32) + 40LL * n + 8);
                if ( v73 )
                {
                  v38 = v70;
                  if ( v70 )
                  {
                    v39 = &v69;
                    do
                    {
                      while ( 1 )
                      {
                        v40 = *(_QWORD *)(v38 + 16);
                        v29 = *(_QWORD *)(v38 + 24);
                        if ( v33 <= *(_DWORD *)(v38 + 32) )
                          break;
                        v38 = *(_QWORD *)(v38 + 24);
                        if ( !v29 )
                          goto LABEL_80;
                      }
                      v39 = (int *)v38;
                      v38 = *(_QWORD *)(v38 + 16);
                    }
                    while ( v40 );
LABEL_80:
                    if ( v39 != &v69 && v33 >= v39[8] )
                      goto LABEL_40;
                  }
                }
                else
                {
                  v34 = v66;
                  v29 = (__int64)&v66[4 * (unsigned int)v67];
                  if ( v66 != (_BYTE *)v29 )
                  {
                    while ( v33 != *v34 )
                    {
                      if ( (_DWORD *)v29 == ++v34 )
                        goto LABEL_61;
                    }
                    if ( (_DWORD *)v29 != v34 )
                      goto LABEL_40;
                  }
                }
LABEL_61:
                v63 = *(_DWORD *)(*(_QWORD *)(v18 + 32) + 40LL * n + 8);
                sub_3361470((__int64)v65, (__int64)&v74, &v63, v29, v30);
              }
            }
            ++*(_DWORD *)(a1 + 6432);
            ++*(_DWORD *)(a1 + 6444);
          }
          else
          {
            ++*(_DWORD *)(a1 + 6436);
          }
          if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 48) + 1016LL))(
                 *(_QWORD *)(a1 + 48),
                 v18,
                 *(_QWORD *)(a1 + 24),
                 *(_QWORD *)(a1 + 16))
            || (*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v64 + 16LL))(v64, v18) )
          {
            goto LABEL_40;
          }
          v21 = *(_BYTE **)(v18 + 32);
          v22 = &v21[40 * (*(_DWORD *)(v18 + 40) & 0xFFFFFF)];
          if ( v21 != v22 )
          {
            v23 = *(_BYTE **)(v18 + 32);
            while ( 1 )
            {
              v24 = v23;
              if ( sub_2DADC00(v23) )
                break;
              v23 += 40;
              if ( v22 == v23 )
                goto LABEL_37;
            }
            while ( v22 != v24 )
            {
              if ( !*v24 && (unsigned int)(*((_DWORD *)v24 + 2) - 1) <= 0x3FFFFFFE )
                goto LABEL_40;
              v25 = v24 + 40;
              if ( v24 + 40 == v22 )
                break;
              while ( 1 )
              {
                v24 = v25;
                if ( sub_2DADC00(v25) )
                  break;
                v25 += 40;
                if ( v22 == v25 )
                  goto LABEL_37;
              }
            }
          }
        }
      }
LABEL_37:
      if ( (*(_BYTE *)v18 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v18 + 44) & 8) != 0 )
          v18 = *(_QWORD *)(v18 + 8);
      }
      v18 = *(_QWORD *)(v18 + 8);
    }
    v62 = *(_DWORD *)(a1 + 6436) > (unsigned int)dword_5040528;
LABEL_40:
    if ( v64 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v64 + 8LL))(v64);
    sub_35E53B0(v78);
    if ( v74 != (unsigned int *)v76 )
      _libc_free((unsigned __int64)v74);
    sub_35E53B0(v70);
    if ( v66 != v68 )
      _libc_free((unsigned __int64)v66);
  }
  else
  {
    return 0;
  }
  return v62;
}
