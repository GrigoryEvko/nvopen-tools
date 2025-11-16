// Function: sub_1473850
// Address: 0x1473850
//
__int64 *__fastcall sub_1473850(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r10d
  __int64 *v8; // r13
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 *v12; // r12
  int v14; // eax
  int v15; // edx
  __int64 v16; // rcx
  _BYTE *v17; // r13
  __int64 v18; // rax
  _BYTE *v19; // r15
  __int64 v20; // r12
  __int64 v21; // rax
  _QWORD *v22; // r14
  _QWORD *v23; // rbx
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned int v26; // eax
  int v27; // r14d
  __int64 v28; // r15
  unsigned int v29; // r13d
  int v30; // edx
  char v31; // r14
  unsigned int v32; // r9d
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // r14
  __int64 v36; // r13
  __int64 v37; // rdx
  int v38; // eax
  __int64 v39; // rsi
  __int64 v40; // rdi
  int v41; // ecx
  unsigned int v42; // edx
  __int64 *v43; // rax
  __int64 v44; // r8
  _QWORD *v45; // rax
  char v46; // dl
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rcx
  int v50; // r8d
  unsigned int v51; // edx
  __int64 *v52; // r13
  __int64 v53; // rdi
  int v54; // eax
  int v55; // edx
  __int64 v56; // rsi
  unsigned int v57; // r13d
  __int64 *v58; // rax
  __int64 v59; // rdi
  int v60; // eax
  int v61; // r9d
  __int64 v62; // r11
  int v63; // ecx
  int v64; // eax
  int v65; // ecx
  __int64 v66; // r10
  __int64 v67; // [rsp+30h] [rbp-1E0h]
  __int64 v69; // [rsp+38h] [rbp-1D8h]
  __int64 v70; // [rsp+38h] [rbp-1D8h]
  __int64 v71; // [rsp+38h] [rbp-1D8h]
  void *v72; // [rsp+40h] [rbp-1D0h] BYREF
  char v73[16]; // [rsp+48h] [rbp-1C8h] BYREF
  __int64 v74; // [rsp+58h] [rbp-1B8h]
  void *v75; // [rsp+70h] [rbp-1A0h] BYREF
  char v76[16]; // [rsp+78h] [rbp-198h] BYREF
  __int64 v77; // [rsp+88h] [rbp-188h]
  __int64 *v78; // [rsp+A0h] [rbp-170h] BYREF
  int v79; // [rsp+A8h] [rbp-168h]
  __int64 v80; // [rsp+C8h] [rbp-148h]
  char v81; // [rsp+D0h] [rbp-140h]
  unsigned __int64 *v82; // [rsp+E0h] [rbp-130h] BYREF
  __int64 v83; // [rsp+E8h] [rbp-128h]
  unsigned __int64 v84[2]; // [rsp+F0h] [rbp-120h] BYREF
  int v85; // [rsp+100h] [rbp-110h]
  __int64 v86; // [rsp+108h] [rbp-108h] BYREF
  char v87; // [rsp+110h] [rbp-100h]
  __int64 *v88; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v89; // [rsp+158h] [rbp-B8h] BYREF
  __int64 v90; // [rsp+160h] [rbp-B0h] BYREF
  _BYTE v91[24]; // [rsp+168h] [rbp-A8h] BYREF
  __int64 v92; // [rsp+180h] [rbp-90h]
  char v93; // [rsp+188h] [rbp-88h]

  v2 = a1 + 528;
  v3 = a1;
  v4 = a2;
  v88 = (__int64 *)a2;
  v5 = *(_DWORD *)(a1 + 552);
  v82 = v84;
  v83 = 0x100000000LL;
  v86 = 0;
  v87 = 0;
  v89 = (__int64)v91;
  v90 = 0x100000000LL;
  v92 = 0;
  v93 = 0;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 528);
    goto LABEL_92;
  }
  v6 = *(_QWORD *)(a1 + 536);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (__int64 *)(v6 + ((unsigned __int64)v9 << 6));
  v11 = *v10;
  if ( v4 != *v10 )
  {
    while ( v11 != -8 )
    {
      if ( !v8 && v11 == -16 )
        v8 = v10;
      v9 = (v5 - 1) & (v7 + v9);
      v10 = (__int64 *)(v6 + ((unsigned __int64)v9 << 6));
      v11 = *v10;
      if ( v4 == *v10 )
        return v10 + 1;
      ++v7;
    }
    if ( !v8 )
      v8 = v10;
    v14 = *(_DWORD *)(a1 + 544);
    ++*(_QWORD *)(a1 + 528);
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v5 )
    {
      v16 = v4;
      if ( v5 - *(_DWORD *)(a1 + 548) - v15 > v5 >> 3 )
        goto LABEL_15;
      sub_1469520(v2, v5);
LABEL_93:
      sub_145FC70(v2, (__int64 *)&v88, &v78);
      v8 = v78;
      v16 = (__int64)v88;
      v15 = *(_DWORD *)(a1 + 544) + 1;
LABEL_15:
      *(_DWORD *)(a1 + 544) = v15;
      if ( *v8 != -8 )
        --*(_DWORD *)(a1 + 548);
      *v8 = v16;
      v8[1] = (__int64)(v8 + 3);
      v8[2] = 0x100000000LL;
      if ( (_DWORD)v90 )
      {
        sub_145E880((__int64)(v8 + 1), (__int64)&v89);
        v18 = (unsigned int)v90;
        v8[6] = v92;
        *((_BYTE *)v8 + 56) = v93;
        v17 = (_BYTE *)(v89 + 24 * v18);
        if ( (_BYTE *)v89 != v17 )
        {
          v19 = (_BYTE *)v89;
          v67 = v4;
          do
          {
            v20 = *((_QWORD *)v17 - 1);
            v17 -= 24;
            if ( v20 )
            {
              *(_QWORD *)v20 = &unk_49EC708;
              v21 = *(unsigned int *)(v20 + 208);
              if ( (_DWORD)v21 )
              {
                v22 = *(_QWORD **)(v20 + 192);
                v23 = &v22[7 * v21];
                do
                {
                  if ( *v22 != -8 && *v22 != -16 )
                  {
                    v24 = v22[1];
                    if ( (_QWORD *)v24 != v22 + 3 )
                      _libc_free(v24);
                  }
                  v22 += 7;
                }
                while ( v23 != v22 );
              }
              j___libc_free_0(*(_QWORD *)(v20 + 192));
              v25 = *(_QWORD *)(v20 + 40);
              if ( v25 != v20 + 56 )
                _libc_free(v25);
              j_j___libc_free_0(v20, 216);
            }
          }
          while ( v19 != v17 );
          v3 = a1;
          v4 = v67;
          v17 = (_BYTE *)v89;
        }
      }
      else
      {
        v8[6] = v92;
        *((_BYTE *)v8 + 56) = v93;
        v17 = (_BYTE *)v89;
      }
      if ( v17 != v91 )
        _libc_free((unsigned __int64)v17);
      sub_1473410((__int64)&v78, v3, v4, 0);
      if ( v79 || !sub_14562D0(v80 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v88 = &v90;
        v89 = 0x1000000000LL;
        sub_1453C80(v4, (__int64)&v88);
        v82 = 0;
        v83 = (__int64)&v86;
        v84[0] = (unsigned __int64)&v86;
        v26 = v89;
        v84[1] = 8;
        v85 = 0;
        if ( (_DWORD)v89 )
        {
          do
          {
            v27 = *(_DWORD *)(v3 + 168);
            v28 = v88[v26 - 1];
            LODWORD(v89) = v26 - 1;
            if ( v27 )
            {
              v69 = *(_QWORD *)(v3 + 152);
              sub_1457D90(&v72, -8, 0);
              sub_1457D90(&v75, -16, 0);
              v29 = ((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4);
              v30 = v27 - 1;
              v31 = 1;
              v32 = v30 & v29;
              v33 = v69 + 48LL * (v30 & v29);
              v34 = *(_QWORD *)(v33 + 24);
              if ( v28 != v34 )
              {
                v62 = v69 + 48LL * (v30 & v29);
                v63 = 1;
                v33 = 0;
                while ( v34 != v74 )
                {
                  if ( v34 != v77 || v33 )
                    v62 = v33;
                  v32 = v30 & (v63 + v32);
                  v33 = v69 + 48LL * v32;
                  v34 = *(_QWORD *)(v33 + 24);
                  if ( v28 == v34 )
                  {
                    v31 = 1;
                    goto LABEL_40;
                  }
                  ++v63;
                  v66 = v62;
                  v62 = v69 + 48LL * v32;
                  v33 = v66;
                }
                v31 = 0;
                if ( !v33 )
                  v33 = v62;
              }
LABEL_40:
              v75 = &unk_49EE2B0;
              if ( v77 != -8 && v77 != 0 && v77 != -16 )
              {
                v70 = v33;
                sub_1649B30(v76);
                v33 = v70;
              }
              v72 = &unk_49EE2B0;
              if ( v74 != 0 && v74 != -8 && v74 != -16 )
              {
                v71 = v33;
                sub_1649B30(v73);
                v33 = v71;
              }
              if ( v31 && v33 != *(_QWORD *)(v3 + 152) + 48LL * *(unsigned int *)(v3 + 168) )
              {
                if ( (v35 = *(_QWORD *)(v33 + 40), *(_BYTE *)(v28 + 16) == 77) && *(_WORD *)(v35 + 24) == 10
                  || (sub_1464220(v3, *(_QWORD *)(v33 + 24)), sub_1459590(v3, v35), *(_BYTE *)(v28 + 16) == 77) )
                {
                  v54 = *(_DWORD *)(v3 + 616);
                  if ( v54 )
                  {
                    v55 = v54 - 1;
                    v56 = *(_QWORD *)(v3 + 600);
                    v57 = (v54 - 1) & v29;
                    v58 = (__int64 *)(v56 + 16LL * v57);
                    v59 = *v58;
                    if ( v28 == *v58 )
                    {
LABEL_74:
                      *v58 = -16;
                      --*(_DWORD *)(v3 + 608);
                      ++*(_DWORD *)(v3 + 612);
                    }
                    else
                    {
                      v64 = 1;
                      while ( v59 != -8 )
                      {
                        v65 = v64 + 1;
                        v57 = v55 & (v64 + v57);
                        v58 = (__int64 *)(v56 + 16LL * v57);
                        v59 = *v58;
                        if ( v28 == *v58 )
                          goto LABEL_74;
                        v64 = v65;
                      }
                    }
                  }
                }
              }
            }
LABEL_51:
            while ( 1 )
            {
              v28 = *(_QWORD *)(v28 + 8);
              if ( !v28 )
                break;
              while ( 1 )
              {
                v36 = sub_1648700(v28);
                if ( *(_BYTE *)(v36 + 16) <= 0x17u )
                  break;
                v37 = *(_QWORD *)(v3 + 64);
                v38 = *(_DWORD *)(v37 + 24);
                if ( !v38 )
                  break;
                v39 = *(_QWORD *)(v36 + 40);
                v40 = *(_QWORD *)(v37 + 8);
                v41 = v38 - 1;
                v42 = (v38 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
                v43 = (__int64 *)(v40 + 16LL * v42);
                v44 = *v43;
                if ( v39 != *v43 )
                {
                  v60 = 1;
                  while ( v44 != -8 )
                  {
                    v61 = v60 + 1;
                    v42 = v41 & (v60 + v42);
                    v43 = (__int64 *)(v40 + 16LL * v42);
                    v44 = *v43;
                    if ( v39 == *v43 )
                      goto LABEL_55;
                    v60 = v61;
                  }
                  goto LABEL_51;
                }
LABEL_55:
                v45 = (_QWORD *)v43[1];
                if ( !v45 )
                  goto LABEL_51;
                if ( (_QWORD *)v4 != v45 )
                {
                  while ( 1 )
                  {
                    v45 = (_QWORD *)*v45;
                    if ( (_QWORD *)v4 == v45 )
                      break;
                    if ( !v45 )
                      goto LABEL_51;
                  }
                }
                sub_1412190((__int64)&v82, v36);
                if ( !v46 )
                  goto LABEL_51;
                v47 = (unsigned int)v89;
                if ( (unsigned int)v89 >= HIDWORD(v89) )
                {
                  sub_16CD150(&v88, &v90, 0, 8);
                  v47 = (unsigned int)v89;
                }
                v88[v47] = v36;
                LODWORD(v89) = v89 + 1;
                v28 = *(_QWORD *)(v28 + 8);
                if ( !v28 )
                  goto LABEL_64;
              }
            }
LABEL_64:
            v26 = v89;
          }
          while ( (_DWORD)v89 );
          if ( v84[0] != v83 )
            _libc_free(v84[0]);
        }
        if ( v88 != &v90 )
          _libc_free((unsigned __int64)v88);
      }
      v48 = *(unsigned int *)(v3 + 552);
      v49 = *(_QWORD *)(v3 + 536);
      if ( (_DWORD)v48 )
      {
        v50 = 1;
        v51 = (v48 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v52 = (__int64 *)(v49 + ((unsigned __int64)v51 << 6));
        v53 = *v52;
        if ( v4 == *v52 )
        {
LABEL_71:
          v12 = v52 + 1;
          sub_145E880((__int64)(v52 + 1), (__int64)&v78);
          v52[6] = v80;
          *((_BYTE *)v52 + 56) = v81;
          sub_1458810((__int64)&v78);
          return v12;
        }
        while ( v53 != -8 )
        {
          v51 = (v48 - 1) & (v50 + v51);
          v52 = (__int64 *)(v49 + ((unsigned __int64)v51 << 6));
          v53 = *v52;
          if ( v4 == *v52 )
            goto LABEL_71;
          ++v50;
        }
      }
      v52 = (__int64 *)(v49 + (v48 << 6));
      goto LABEL_71;
    }
LABEL_92:
    sub_1469520(v2, 2 * v5);
    goto LABEL_93;
  }
  return v10 + 1;
}
