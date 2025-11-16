// Function: sub_25AEA80
// Address: 0x25aea80
//
__int64 __fastcall sub_25AEA80(__int64 a1)
{
  __int64 v1; // r15
  unsigned __int64 *v3; // rsi
  unsigned __int64 *v4; // rax
  unsigned __int64 *v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  const void *v10; // r12
  const void *v11; // r14
  __int64 v12; // rbx
  unsigned __int64 v13; // rax
  char v14; // r13
  __int64 v15; // r12
  __int64 v16; // rax
  char v17; // r13
  unsigned int v18; // r14d
  unsigned int v19; // r14d
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rcx
  __int64 v23; // rbx
  __int64 v24; // rax
  unsigned __int8 *v25; // r12
  int v26; // eax
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdx
  unsigned __int64 v30; // r13
  unsigned __int8 *v31; // r14
  __int64 v32; // r15
  _QWORD *i; // rax
  bool v34; // zf
  int v35; // r13d
  __int64 v36; // r14
  unsigned __int64 *v37; // r9
  unsigned __int64 v38; // rsi
  int v39; // edi
  __int64 v40; // rax
  unsigned __int8 *v41; // rax
  __int64 v42; // r15
  int v43; // ecx
  unsigned __int8 *v44; // r13
  __int16 v45; // dx
  unsigned __int64 v46; // r13
  unsigned __int64 *v47; // r14
  unsigned __int64 v48; // rdi
  unsigned __int64 v49; // rax
  int v50; // r15d
  __int64 v51; // rax
  __int64 v52; // r9
  __int64 v53; // rdx
  unsigned __int64 v54; // r8
  unsigned int v55; // eax
  unsigned __int64 *v56; // rcx
  unsigned __int64 v57; // r15
  unsigned __int64 v58; // r14
  _QWORD *v59; // rax
  __int64 v60; // r14
  unsigned __int64 *v61; // r9
  unsigned __int64 v62; // rcx
  int v63; // esi
  __int64 v64; // rax
  __int64 v65; // rsi
  unsigned __int8 *v66; // rax
  int v67; // ecx
  __int64 *v68; // rax
  __int64 *v69; // rsi
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r12
  __int64 v73; // rdx
  __int64 v74; // rbx
  unsigned __int8 *j; // r13
  unsigned __int8 *v76; // rsi
  unsigned int *v77; // r12
  unsigned int *v78; // rbx
  __int64 v79; // rdx
  __int64 v80; // rsi
  __int64 v81; // rdx
  signed __int64 v82; // [rsp+8h] [rbp-1C0h]
  size_t n; // [rsp+10h] [rbp-1B8h]
  __int64 v84; // [rsp+18h] [rbp-1B0h]
  void *dest; // [rsp+20h] [rbp-1A8h]
  int v86; // [rsp+2Ch] [rbp-19Ch]
  signed __int64 v88; // [rsp+40h] [rbp-188h]
  __int64 v89; // [rsp+50h] [rbp-178h]
  __int64 v90; // [rsp+58h] [rbp-170h]
  unsigned int v91; // [rsp+68h] [rbp-160h]
  unsigned int v92; // [rsp+6Ch] [rbp-15Ch]
  __int64 v93; // [rsp+70h] [rbp-158h]
  __int64 v94; // [rsp+70h] [rbp-158h]
  __int64 v95; // [rsp+78h] [rbp-150h]
  unsigned __int64 *v96; // [rsp+78h] [rbp-150h]
  __int64 v97; // [rsp+78h] [rbp-150h]
  __int64 v98; // [rsp+78h] [rbp-150h]
  unsigned __int64 v99; // [rsp+80h] [rbp-148h]
  __int64 v100; // [rsp+98h] [rbp-130h]
  __int64 v101; // [rsp+A8h] [rbp-120h]
  int v102; // [rsp+B0h] [rbp-118h]
  unsigned int v103; // [rsp+B0h] [rbp-118h]
  __int64 v104; // [rsp+E8h] [rbp-E0h]
  __int64 *v105; // [rsp+F0h] [rbp-D8h]
  unsigned __int64 *v106; // [rsp+100h] [rbp-C8h]
  unsigned __int64 v107; // [rsp+110h] [rbp-B8h] BYREF
  int v108[8]; // [rsp+118h] [rbp-B0h] BYREF
  __int16 v109; // [rsp+138h] [rbp-90h]
  unsigned __int64 *v110; // [rsp+148h] [rbp-80h] BYREF
  __int64 v111; // [rsp+150h] [rbp-78h]
  _BYTE v112[16]; // [rsp+158h] [rbp-70h] BYREF
  __int16 v113; // [rsp+168h] [rbp-60h]

  v1 = a1;
  if ( (unsigned __int8)sub_B2DDD0(a1, 0, 0, 1, 0, 0, 0) || (unsigned __int8)sub_B2D610(a1, 20) )
    return 0;
  v3 = *(unsigned __int64 **)(a1 + 80);
  v106 = (unsigned __int64 *)(a1 + 72);
  if ( v3 != (unsigned __int64 *)(a1 + 72) )
  {
    while ( 1 )
    {
      if ( !v3 )
        BUG();
      v4 = (unsigned __int64 *)v3[4];
      v5 = v3 + 3;
      if ( v4 != v3 + 3 )
        break;
LABEL_18:
      v3 = (unsigned __int64 *)v3[1];
      if ( v106 == v3 )
        goto LABEL_19;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v4 )
          BUG();
        if ( *((_BYTE *)v4 - 24) == 85 )
        {
          if ( (*((_WORD *)v4 - 11) & 3) == 2 )
            return 0;
          v6 = *(v4 - 7);
          if ( v6 )
          {
            if ( !*(_BYTE *)v6 && *(_QWORD *)(v6 + 24) == v4[7] && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
              break;
          }
        }
        v4 = (unsigned __int64 *)v4[1];
        if ( v5 == v4 )
          goto LABEL_18;
      }
      if ( *(_DWORD *)(v6 + 36) == 375 )
        return 0;
      v4 = (unsigned __int64 *)v4[1];
      if ( v5 == v4 )
        goto LABEL_18;
    }
  }
LABEL_19:
  v7 = *(_QWORD *)(a1 + 24);
  v8 = *(_QWORD *)(v7 + 16);
  v9 = 8LL * *(unsigned int *)(v7 + 12);
  v10 = (const void *)(v8 + 8);
  v11 = (const void *)(v8 + v9);
  n = v9 - 8;
  v12 = (v9 - 8) >> 3;
  if ( (unsigned __int64)(v9 - 8) > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  dest = 0;
  if ( v12 )
    dest = (void *)sub_22077B0(n);
  if ( v11 != v10 )
    memcpy(dest, v10, n);
  v13 = sub_BCF480(**(__int64 ***)(v7 + 16), dest, v12, 0);
  v14 = *(_BYTE *)(a1 + 32);
  v15 = v13;
  v16 = *(_QWORD *)(a1 + 8);
  v113 = 257;
  v86 = v12;
  v17 = v14 & 0xF;
  v18 = *(_DWORD *)(v16 + 8);
  v104 = sub_BD2DA0(136);
  v19 = v18 >> 8;
  if ( v104 )
    sub_B2C3B0(v104, v15, v17, v19, (__int64)&v110, 0);
  sub_B2EC90(v104, a1);
  sub_B2F990(v104, *(_QWORD *)(a1 + 48), v20, v21);
  sub_BA8540(*(_QWORD *)(a1 + 40) + 24LL, v104);
  v22 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(v104 + 64) = a1 + 56;
  v22 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v104 + 56) = v22 | *(_QWORD *)(v104 + 56) & 7LL;
  *(_QWORD *)(v22 + 8) = v104 + 56;
  *(_QWORD *)(a1 + 56) = *(_QWORD *)(a1 + 56) & 7LL | (v104 + 56);
  sub_BD6B90((unsigned __int8 *)v104, (unsigned __int8 *)a1);
  *(_BYTE *)(v104 + 128) = *(_BYTE *)(a1 + 128);
  if ( *(_QWORD *)(a1 + 16) )
  {
    v99 = (unsigned int)v12;
    v84 = 32LL * (unsigned int)v12;
    v88 = 8LL * (unsigned int)v12;
    v23 = *(_QWORD *)(a1 + 16);
    v100 = 0;
    v105 = 0;
    v82 = 8 * ((unsigned __int64)(v84 - 32) >> 5) + 8;
    do
    {
      v24 = v23;
      v23 = *(_QWORD *)(v23 + 8);
      v25 = *(unsigned __int8 **)(v24 + 24);
      v26 = *v25;
      if ( (unsigned __int8)v26 > 0x1Cu )
      {
        v27 = (unsigned int)(v26 - 34);
        if ( (unsigned __int8)v27 <= 0x33u )
        {
          v28 = 0x8000000000041LL;
          if ( _bittest64(&v28, v27) )
          {
            v29 = *((_DWORD *)v25 + 1) & 0x7FFFFFF;
            v30 = (unsigned __int64)&v25[-32 * v29];
            v31 = &v25[32 * (v99 - v29)];
            if ( v99 <= (v100 - (__int64)v105) >> 3 )
            {
              if ( v99 )
              {
                if ( (unsigned __int8 *)v30 == v31 )
                {
                  v102 = 0;
                  v101 = 0;
                }
                else
                {
                  v68 = v105;
                  do
                  {
                    if ( v68 )
                      *v68 = *(_QWORD *)v30;
                    v30 += 32LL;
                    ++v68;
                  }
                  while ( v31 != (unsigned __int8 *)v30 );
                  v101 = v82 >> 3;
                  v102 = v82 >> 3;
                }
              }
              else
              {
                if ( v84 )
                {
                  v49 = 0;
                  do
                  {
                    v105[v49 / 8] = *(_QWORD *)(v30 + 4 * v49);
                    v49 += 8LL;
                  }
                  while ( v49 );
                }
                v101 = 0;
                v102 = 0;
              }
            }
            else
            {
              v32 = 0;
              if ( v99 )
                v32 = sub_22077B0(v88);
              for ( i = (_QWORD *)v32; v31 != (unsigned __int8 *)v30; ++i )
              {
                if ( i )
                  *i = *(_QWORD *)v30;
                v30 += 32LL;
              }
              if ( v105 )
                j_j___libc_free_0((unsigned __int64)v105);
              v105 = (__int64 *)v32;
              v100 = v32 + v88;
              v101 = v88 >> 3;
              v102 = v88 >> 3;
            }
            v107 = *((_QWORD *)v25 + 9);
            if ( v107 )
            {
              v110 = (unsigned __int64 *)v112;
              v111 = 0x800000000LL;
              if ( v86 )
              {
                v50 = 0;
                do
                {
                  v51 = sub_A744E0(&v107, v50);
                  v53 = (unsigned int)v111;
                  v54 = (unsigned int)v111 + 1LL;
                  if ( v54 > HIDWORD(v111) )
                  {
                    v98 = v51;
                    sub_C8D5F0((__int64)&v110, v112, (unsigned int)v111 + 1LL, 8u, v54, v52);
                    v53 = (unsigned int)v111;
                    v51 = v98;
                  }
                  ++v50;
                  v110[v53] = v51;
                  v55 = v111 + 1;
                  LODWORD(v111) = v111 + 1;
                }
                while ( v86 != v50 );
                v56 = v110;
                v30 = v55;
              }
              else
              {
                v56 = (unsigned __int64 *)v112;
                v30 = 0;
              }
              v96 = v56;
              v57 = sub_A74610(&v107);
              v58 = sub_A74680(&v107);
              v59 = (_QWORD *)sub_B2BE50(a1);
              v107 = sub_A78180(v59, v58, v57, v96, v30);
              if ( v110 != (unsigned __int64 *)v112 )
                _libc_free((unsigned __int64)v110);
            }
            v110 = (unsigned __int64 *)v112;
            v111 = 0x100000000LL;
            sub_B56970((__int64)v25, (__int64)&v110);
            v34 = *v25 == 34;
            v109 = 257;
            if ( v34 )
            {
              v95 = *((_QWORD *)v25 - 8);
              v35 = v111;
              v93 = *((_QWORD *)v25 - 12);
              v36 = *(_QWORD *)(v104 + 24);
              v37 = &v110[7 * (unsigned int)v111];
              if ( v110 == v37 )
              {
                v39 = 0;
              }
              else
              {
                v38 = (unsigned __int64)v110;
                v39 = 0;
                do
                {
                  v40 = *(_QWORD *)(v38 + 40) - *(_QWORD *)(v38 + 32);
                  v38 += 56LL;
                  v39 += v40 >> 3;
                }
                while ( v37 != (unsigned __int64 *)v38 );
              }
              v89 = (unsigned int)v111;
              LOBYTE(v35) = 16 * (_DWORD)v111 != 0;
              v90 = (__int64)v110;
              v103 = v39 + v102 + 3;
              v41 = (unsigned __int8 *)sub_BD2CC0(88, v103 | ((unsigned __int64)(unsigned int)(16 * v111) << 32));
              v42 = (__int64)v41;
              if ( v41 )
              {
                v43 = (v35 << 28) | v103 & 0x7FFFFFF;
                v44 = v41;
                v92 = v92 & 0xE0000000 | v43;
                sub_B44260((__int64)v41, **(_QWORD **)(v36 + 16), 5, v92, (__int64)(v25 + 24), 0);
                *(_QWORD *)(v42 + 72) = 0;
                sub_B4A9C0(v42, v36, v104, v93, v95, (__int64)v108, v105, v101, v90, v89);
              }
              else
              {
                v44 = 0;
              }
              v45 = *(_WORD *)(v42 + 2);
            }
            else
            {
              v60 = *(_QWORD *)(v104 + 24);
              v61 = &v110[7 * (unsigned int)v111];
              if ( v110 == v61 )
              {
                v63 = 0;
              }
              else
              {
                v62 = (unsigned __int64)v110;
                v63 = 0;
                do
                {
                  v64 = *(_QWORD *)(v62 + 40) - *(_QWORD *)(v62 + 32);
                  v62 += 56LL;
                  v63 += v64 >> 3;
                }
                while ( v61 != (unsigned __int64 *)v62 );
              }
              LOBYTE(v30) = 16 * (_DWORD)v111 != 0;
              v94 = (unsigned int)v111;
              v65 = (unsigned int)(v63 + v102 + 1);
              v97 = (__int64)v110;
              v66 = (unsigned __int8 *)sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v111) << 32) | v65);
              v42 = (__int64)v66;
              if ( v66 )
              {
                v67 = ((_DWORD)v30 << 28) | v65 & 0x7FFFFFF;
                v44 = v66;
                v91 = v91 & 0xE0000000 | v67;
                sub_B44260((__int64)v66, **(_QWORD **)(v60 + 16), 56, v91, (__int64)(v25 + 24), 0);
                *(_QWORD *)(v42 + 72) = 0;
                sub_B4A290(v42, v60, v104, v105, v101, (__int64)v108, v97, v94);
              }
              else
              {
                v44 = 0;
              }
              v45 = *(_WORD *)(v42 + 2) & 0xFFFC | *((_WORD *)v25 + 1) & 3;
              *(_WORD *)(v42 + 2) = v45;
            }
            *(_WORD *)(v42 + 2) = *((_WORD *)v25 + 1) & 0xFFC | v45 & 0xF003;
            *(_QWORD *)(v42 + 72) = v107;
            *(_QWORD *)v108 = 2;
            sub_B47C00((__int64)v44, (__int64)v25, v108, 2);
            if ( *((_QWORD *)v25 + 2) )
              sub_BD84D0((__int64)v25, v42);
            sub_BD6B90(v44, v25);
            sub_B43D60(v25);
            v46 = (unsigned __int64)v110;
            v47 = &v110[7 * (unsigned int)v111];
            if ( v110 != v47 )
            {
              do
              {
                v48 = *(v47 - 3);
                v47 -= 7;
                if ( v48 )
                  j_j___libc_free_0(v48);
                if ( (unsigned __int64 *)*v47 != v47 + 2 )
                  j_j___libc_free_0(*v47);
              }
              while ( (unsigned __int64 *)v46 != v47 );
              v47 = v110;
            }
            if ( v47 != (unsigned __int64 *)v112 )
              _libc_free((unsigned __int64)v47);
          }
        }
      }
    }
    while ( v23 );
    v1 = a1;
  }
  else
  {
    v105 = 0;
  }
  v69 = *(__int64 **)(v104 + 80);
  sub_B2C300(v104, v69, v1, *(unsigned __int64 **)(v1 + 80), v106);
  if ( (*(_BYTE *)(v1 + 2) & 1) != 0 )
  {
    sub_B2C6D0(v1, (__int64)v69, v70, v71);
    v72 = *(_QWORD *)(v1 + 96);
    if ( (*(_BYTE *)(v1 + 2) & 1) != 0 )
      sub_B2C6D0(v1, (__int64)v69, v81, v71);
    v73 = *(_QWORD *)(v1 + 96);
  }
  else
  {
    v72 = *(_QWORD *)(v1 + 96);
    v73 = v72;
  }
  v74 = v73 + 40LL * *(_QWORD *)(v1 + 104);
  if ( (*(_BYTE *)(v104 + 2) & 1) != 0 )
    sub_B2C6D0(v104, (__int64)v69, v73, v71);
  for ( j = *(unsigned __int8 **)(v104 + 96); v72 != v74; j += 40 )
  {
    sub_BD84D0(v72, (__int64)j);
    v76 = (unsigned __int8 *)v72;
    v72 += 40;
    sub_BD6B90(j, v76);
  }
  v110 = (unsigned __int64 *)v112;
  v111 = 0x100000000LL;
  sub_B9A9D0(v1, (__int64)&v110);
  v77 = (unsigned int *)v110;
  v78 = (unsigned int *)&v110[2 * (unsigned int)v111];
  if ( v110 != (unsigned __int64 *)v78 )
  {
    do
    {
      v79 = *((_QWORD *)v77 + 1);
      v80 = *v77;
      v77 += 4;
      sub_B994D0(v104, v80, v79);
    }
    while ( v78 != v77 );
  }
  sub_BD84D0(v1, v104);
  sub_AD0030(v104);
  sub_B2E860((_QWORD *)v1);
  if ( v110 != (unsigned __int64 *)v112 )
    _libc_free((unsigned __int64)v110);
  if ( v105 )
    j_j___libc_free_0((unsigned __int64)v105);
  if ( dest )
    j_j___libc_free_0((unsigned __int64)dest);
  return 1;
}
