// Function: sub_1F16160
// Address: 0x1f16160
//
void __fastcall sub_1F16160(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rax
  unsigned __int64 v3; // r9
  __int64 *v5; // r13
  __int64 v6; // r14
  __int64 v7; // rsi
  int v8; // r8d
  int v9; // edx
  __int64 *v10; // rdi
  unsigned int v11; // ecx
  unsigned int v12; // eax
  __int64 *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 v17; // r8
  unsigned __int64 v18; // rcx
  int v19; // r11d
  unsigned int v20; // edx
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // r15
  __int64 v25; // r14
  __int64 v26; // r14
  __int64 v27; // r15
  __int64 *v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // r9
  __int64 *v31; // r15
  __int64 v32; // r12
  __int64 v33; // rsi
  int v34; // ecx
  unsigned int v35; // r8d
  __int64 *v36; // rdi
  int v37; // edx
  unsigned int v38; // eax
  __int64 *v39; // rax
  __int64 v40; // r8
  unsigned __int64 v41; // rcx
  int v42; // r11d
  unsigned int v43; // edx
  __int64 v44; // r10
  __int64 v45; // rax
  __int64 v46; // r14
  __int64 i; // r10
  __int64 v48; // r12
  int v49; // r9d
  __int64 v50; // rax
  _QWORD *v51; // rcx
  __int64 v52; // r9
  _QWORD *v53; // rbx
  _QWORD *v54; // r12
  unsigned int v55; // r14d
  __int64 v56; // rsi
  __int64 v57; // rdx
  unsigned int v58; // r12d
  __int64 v59; // rsi
  __int64 v60; // rsi
  __int64 v61; // r9
  __int64 v62; // rax
  _QWORD *v63; // rdi
  _QWORD *v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // rdx
  _QWORD *v67; // rdi
  _QWORD *v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // [rsp+8h] [rbp-368h]
  __int64 v71; // [rsp+18h] [rbp-358h]
  int v72; // [rsp+20h] [rbp-350h]
  __int64 v73; // [rsp+28h] [rbp-348h]
  __int64 v74; // [rsp+40h] [rbp-330h]
  __int64 v75; // [rsp+40h] [rbp-330h]
  int v76; // [rsp+40h] [rbp-330h]
  bool v77; // [rsp+48h] [rbp-328h]
  __int64 v78; // [rsp+48h] [rbp-328h]
  __int64 v79; // [rsp+48h] [rbp-328h]
  __int64 v80; // [rsp+48h] [rbp-328h]
  int v81; // [rsp+60h] [rbp-310h]
  __int64 v82; // [rsp+60h] [rbp-310h]
  __int64 v83; // [rsp+60h] [rbp-310h]
  _QWORD *v84; // [rsp+60h] [rbp-310h]
  _QWORD *v85; // [rsp+60h] [rbp-310h]
  unsigned __int64 v86; // [rsp+68h] [rbp-308h]
  __int64 v87; // [rsp+68h] [rbp-308h]
  __int64 *v88; // [rsp+70h] [rbp-300h] BYREF
  __int64 v89; // [rsp+78h] [rbp-2F8h]
  _BYTE v90[32]; // [rsp+80h] [rbp-2F0h] BYREF
  __m128i v91; // [rsp+A0h] [rbp-2D0h] BYREF
  __int64 v92; // [rsp+B0h] [rbp-2C0h]
  __int64 v93; // [rsp+B8h] [rbp-2B8h]
  __int64 v94; // [rsp+C0h] [rbp-2B0h]
  unsigned __int64 v95; // [rsp+C8h] [rbp-2A8h]
  __int64 v96; // [rsp+D0h] [rbp-2A0h]
  int v97; // [rsp+D8h] [rbp-298h]
  __int64 v98; // [rsp+E0h] [rbp-290h]
  _QWORD *v99; // [rsp+E8h] [rbp-288h]
  __int64 v100; // [rsp+F0h] [rbp-280h]
  unsigned int v101; // [rsp+F8h] [rbp-278h]
  _QWORD *v102; // [rsp+100h] [rbp-270h]
  __int64 v103; // [rsp+108h] [rbp-268h]
  _QWORD v104[3]; // [rsp+110h] [rbp-260h] BYREF
  _BYTE *v105; // [rsp+128h] [rbp-248h]
  __int64 v106; // [rsp+130h] [rbp-240h]
  _BYTE v107[568]; // [rsp+138h] [rbp-238h] BYREF

  v1 = a1;
  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v3 = *(_QWORD *)(v2 + 64);
  v74 = v2;
  v86 = v3 + 8LL * *(unsigned int *)(v2 + 72);
  if ( v3 == v86 )
    goto LABEL_23;
  v5 = *(__int64 **)(v2 + 64);
  do
  {
    while ( 1 )
    {
      v6 = *v5;
      v7 = *(_QWORD *)(*v5 + 8);
      if ( (v7 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v7 & 6) != 0 )
        goto LABEL_21;
      v8 = *(_DWORD *)(a1 + 388);
      if ( !v8 )
      {
LABEL_17:
        v77 = 0;
        v9 = 0;
        goto LABEL_18;
      }
      v9 = *(_DWORD *)(a1 + 384);
      v10 = (__int64 *)(a1 + 200);
      v11 = *(_DWORD *)((*(_QWORD *)(*v5 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24);
      v12 = *(_DWORD *)((*(_QWORD *)(a1 + 200) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(a1 + 200) >> 1) & 3;
      if ( v9 )
      {
        if ( v12 > v11 )
          goto LABEL_17;
        v13 = (__int64 *)(a1 + 8LL * (unsigned int)(v8 - 1) + 296);
      }
      else
      {
        if ( v12 > v11 )
          goto LABEL_17;
        v13 = &v10[2 * (unsigned int)(v8 - 1) + 1];
      }
      if ( (*(_DWORD *)((*v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v13 >> 1) & 3) <= v11 )
        goto LABEL_17;
      if ( v9 )
      {
        v9 = sub_1F15FF0((__int64)v10, v7, 0);
        v77 = v9 != 0;
      }
      else
      {
        if ( (*(_DWORD *)((*(_QWORD *)(a1 + 208) & 0xFFFFFFFFFFFFFFF8LL) + 24)
            | (unsigned int)(*(__int64 *)(a1 + 208) >> 1) & 3) > v11 )
        {
          v14 = 0;
        }
        else
        {
          LODWORD(v14) = 0;
          do
          {
            v14 = (unsigned int)(v14 + 1);
            v15 = v10[2 * v14 + 1];
            v16 = v15 >> 1;
            v3 = v15 & 0xFFFFFFFFFFFFFFF8LL;
          }
          while ( (*(_DWORD *)(v3 + 24) | (unsigned int)(v16 & 3)) <= v11 );
          v10 += 2 * v14;
        }
        v77 = 0;
        if ( (*(_DWORD *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v10 >> 1) & 3) <= v11 )
        {
          v9 = *(_DWORD *)(a1 + 4 * v14 + 344);
          v77 = v9 != 0;
        }
      }
LABEL_18:
      v17 = *(_QWORD *)(a1 + 16);
      v18 = *(unsigned int *)(v17 + 408);
      v19 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                      + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL) + v9));
      v20 = v19 & 0x7FFFFFFF;
      v21 = v19 & 0x7FFFFFFF;
      v22 = 8 * v21;
      if ( (v19 & 0x7FFFFFFFu) >= (unsigned int)v18 || (v23 = *(_QWORD *)(*(_QWORD *)(v17 + 400) + 8LL * v20)) == 0 )
      {
        v58 = v20 + 1;
        if ( (unsigned int)v18 >= v20 + 1 )
          goto LABEL_69;
        v66 = v58;
        if ( v58 < v18 )
        {
          *(_DWORD *)(v17 + 408) = v58;
          goto LABEL_69;
        }
        if ( v58 <= v18 )
        {
LABEL_69:
          v59 = *(_QWORD *)(v17 + 400);
        }
        else
        {
          if ( v58 > (unsigned __int64)*(unsigned int *)(v17 + 412) )
          {
            v71 = 8LL * (v19 & 0x7FFFFFFF);
            v72 = v19;
            v73 = *(_QWORD *)(a1 + 16);
            sub_16CD150(v17 + 400, (const void *)(v17 + 416), v58, 8, v17, v3);
            v17 = v73;
            v22 = v71;
            v19 = v72;
            v66 = v58;
            v18 = *(unsigned int *)(v73 + 408);
          }
          v59 = *(_QWORD *)(v17 + 400);
          v67 = (_QWORD *)(v59 + 8 * v66);
          v68 = (_QWORD *)(v59 + 8 * v18);
          v69 = *(_QWORD *)(v17 + 416);
          if ( v67 != v68 )
          {
            do
              *v68++ = v69;
            while ( v67 != v68 );
            v59 = *(_QWORD *)(v17 + 400);
          }
          *(_DWORD *)(v17 + 408) = v58;
        }
        v85 = (_QWORD *)v17;
        *(_QWORD *)(v59 + v22) = sub_1DBA290(v19);
        v23 = *(_QWORD *)(v85[50] + 8 * v21);
        sub_1DBB110(v85, v23);
        v17 = *(_QWORD *)(a1 + 16);
      }
      v24 = *(_QWORD *)(v6 + 8);
      v81 = *(_DWORD *)(a1 + 84);
      v25 = sub_1DA9310(*(_QWORD *)(v17 + 272), v24);
      if ( !(unsigned __int8)sub_1F13900(v24, v23) )
        break;
LABEL_21:
      if ( (__int64 *)v86 == ++v5 )
        goto LABEL_22;
    }
    ++v5;
    sub_1F156C0(a1, v25, (_QWORD *)(a1 + 664LL * (v77 & (unsigned __int8)(v81 != 0)) + 432), v23, -1, v3, 0, 0);
  }
  while ( (__int64 *)v86 != v5 );
LABEL_22:
  v1 = a1;
LABEL_23:
  v103 = 0;
  v88 = (__int64 *)v90;
  v89 = 0x400000000LL;
  v102 = v104;
  v105 = v107;
  v106 = 0x1000000000LL;
  v104[0] = 0;
  v104[1] = 0;
  v26 = *(_QWORD *)(v74 + 104);
  v91 = 0u;
  v27 = v26;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  if ( !v26 )
    goto LABEL_49;
  while ( 2 )
  {
    v28 = *(__int64 **)(v27 + 64);
    v29 = *(unsigned int *)(v27 + 72);
    v30 = (unsigned __int64)&v28[v29];
    if ( v28 == (__int64 *)v30 )
      goto LABEL_44;
    v87 = v27;
    v31 = &v28[v29];
    while ( 2 )
    {
      while ( 2 )
      {
        v32 = *v28;
        v33 = *(_QWORD *)(*v28 + 8);
        if ( (v33 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v33 & 6) != 0 )
        {
LABEL_26:
          if ( v31 == ++v28 )
            goto LABEL_43;
          continue;
        }
        break;
      }
      v34 = *(_DWORD *)(v1 + 388);
      if ( !v34 )
        goto LABEL_58;
      v35 = *(_DWORD *)((*(_QWORD *)(*v28 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24);
      v36 = (__int64 *)(v1 + 200);
      v37 = *(_DWORD *)(v1 + 384);
      v38 = (*(__int64 *)(v1 + 200) >> 1) & 3 | *(_DWORD *)((*(_QWORD *)(v1 + 200) & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( !v37 )
      {
        if ( v38 <= v35 )
        {
          v39 = &v36[2 * (unsigned int)(v34 - 1) + 1];
          goto LABEL_33;
        }
        goto LABEL_58;
      }
      if ( v38 > v35 )
        goto LABEL_58;
      v39 = (__int64 *)(v1 + 8LL * (unsigned int)(v34 - 1) + 296);
LABEL_33:
      if ( (*(_DWORD *)((*v39 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v39 >> 1) & 3) <= v35 )
      {
LABEL_58:
        v37 = 0;
      }
      else if ( v37 )
      {
        v37 = sub_1F15FF0((__int64)v36, v33, 0);
      }
      else
      {
        if ( (*(_DWORD *)((*(_QWORD *)(v1 + 208) & 0xFFFFFFFFFFFFFFF8LL) + 24)
            | (unsigned int)(*(__int64 *)(v1 + 208) >> 1) & 3) > v35 )
        {
          v60 = 0;
        }
        else
        {
          LODWORD(v60) = 0;
          do
          {
            v60 = (unsigned int)(v60 + 1);
            v61 = v36[2 * v60 + 1];
            v62 = v61 >> 1;
            v30 = v61 & 0xFFFFFFFFFFFFFFF8LL;
          }
          while ( (*(_DWORD *)(v30 + 24) | (unsigned int)(v62 & 3)) <= v35 );
          v36 += 2 * v60;
        }
        if ( (*(_DWORD *)((*v36 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v36 >> 1) & 3) <= v35 )
          v37 = *(_DWORD *)(v1 + 4 * v60 + 344);
      }
      v40 = *(_QWORD *)(v1 + 16);
      v41 = *(unsigned int *)(v40 + 408);
      v42 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(v1 + 72) + 16LL)
                      + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(v1 + 72) + 64LL) + v37));
      v43 = v42 & 0x7FFFFFFF;
      v44 = v42 & 0x7FFFFFFF;
      v45 = 8 * v44;
      if ( (v42 & 0x7FFFFFFFu) >= (unsigned int)v41 || (v46 = *(_QWORD *)(*(_QWORD *)(v40 + 400) + 8LL * v43)) == 0 )
      {
        v55 = v43 + 1;
        if ( (unsigned int)v41 < v43 + 1 )
        {
          v57 = v55;
          if ( v55 >= v41 )
          {
            if ( v55 > v41 )
            {
              if ( v55 > (unsigned __int64)*(unsigned int *)(v40 + 412) )
              {
                v70 = v42 & 0x7FFFFFFF;
                v76 = v42;
                v80 = *(_QWORD *)(v1 + 16);
                sub_16CD150(v40 + 400, (const void *)(v40 + 416), v55, 8, v40, v30);
                v40 = v80;
                v44 = v70;
                v45 = 8 * v70;
                v42 = v76;
                v41 = *(unsigned int *)(v80 + 408);
                v57 = v55;
              }
              v56 = *(_QWORD *)(v40 + 400);
              v63 = (_QWORD *)(v56 + 8 * v57);
              v64 = (_QWORD *)(v56 + 8 * v41);
              v65 = *(_QWORD *)(v40 + 416);
              if ( v63 != v64 )
              {
                do
                  *v64++ = v65;
                while ( v63 != v64 );
                v56 = *(_QWORD *)(v40 + 400);
              }
              *(_DWORD *)(v40 + 408) = v55;
              goto LABEL_61;
            }
          }
          else
          {
            *(_DWORD *)(v40 + 408) = v55;
          }
        }
        v56 = *(_QWORD *)(v40 + 400);
LABEL_61:
        v79 = v44;
        v84 = (_QWORD *)v40;
        *(_QWORD *)(v56 + v45) = sub_1DBA290(v42);
        v46 = *(_QWORD *)(v84[50] + 8 * v79);
        sub_1DBB110(v84, v46);
      }
      for ( i = *(_QWORD *)(v46 + 104); *(_DWORD *)(v87 + 112) != *(_DWORD *)(i + 112); i = *(_QWORD *)(i + 104) )
        ;
      v82 = i;
      if ( (unsigned __int8)sub_1F13900(*(_QWORD *)(v32 + 8), i) )
        goto LABEL_26;
      ++v28;
      v75 = v82;
      v78 = *(_QWORD *)(v1 + 16);
      v83 = *(_QWORD *)(v78 + 272);
      v48 = sub_1DA9310(v83, *(_QWORD *)(v32 + 8));
      sub_1DC3BD0(&v91, *(_QWORD *)(*(_QWORD *)(v1 + 24) + 256LL), v83, *(_QWORD *)(v1 + 40), v78 + 296, v49);
      v50 = *(_QWORD *)(v1 + 16);
      v51 = *(_QWORD **)(v1 + 32);
      LODWORD(v89) = 0;
      sub_1DB4D80(v46, (__int64)&v88, *(_DWORD *)(v87 + 112), v51, *(_QWORD *)(v50 + 272));
      sub_1F156C0(v1, v48, &v91, v75, *(_DWORD *)(v87 + 112), v52, v88, (unsigned int)v89);
      if ( v31 != v28 )
        continue;
      break;
    }
LABEL_43:
    v27 = v87;
LABEL_44:
    v27 = *(_QWORD *)(v27 + 104);
    if ( v27 )
      continue;
    break;
  }
  if ( v105 != v107 )
    _libc_free((unsigned __int64)v105);
  if ( v102 != v104 )
    _libc_free((unsigned __int64)v102);
LABEL_49:
  if ( v101 )
  {
    v53 = v99;
    v54 = &v99[7 * v101];
    do
    {
      if ( *v53 != -16 && *v53 != -8 )
      {
        _libc_free(v53[4]);
        _libc_free(v53[1]);
      }
      v53 += 7;
    }
    while ( v54 != v53 );
  }
  j___libc_free_0(v99);
  _libc_free(v95);
  if ( v88 != (__int64 *)v90 )
    _libc_free((unsigned __int64)v88);
}
