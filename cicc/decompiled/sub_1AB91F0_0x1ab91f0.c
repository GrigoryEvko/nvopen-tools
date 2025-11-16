// Function: sub_1AB91F0
// Address: 0x1ab91f0
//
__int64 __fastcall sub_1AB91F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v10; // r12
  _QWORD *v11; // rax
  _BYTE *v12; // rsi
  _QWORD *v13; // rdi
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v18; // r12
  __int64 v19; // r15
  _QWORD *v20; // rdi
  __int64 v21; // rax
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rax
  __int64 *v25; // r12
  __int64 v26; // rax
  __int64 v27; // rsi
  unsigned int v28; // ecx
  __int64 *v29; // rdx
  __int64 v30; // r9
  __int64 v31; // r15
  _QWORD *v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rdi
  __int64 v35; // rax
  int v36; // r9d
  unsigned int v37; // r10d
  __int64 *v38; // r8
  __int64 v39; // rdx
  __int64 *v40; // rax
  __int64 v41; // r15
  unsigned int v42; // ecx
  __int64 *v43; // rdx
  __int64 v44; // r11
  __int64 v45; // rcx
  __int64 v46; // r9
  char *v47; // rdx
  char *v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rsi
  char *v51; // rax
  _BYTE *v52; // rsi
  _QWORD *v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 *v57; // r12
  __int64 *v58; // r15
  __int64 v59; // r13
  __int64 v60; // rcx
  unsigned __int64 *v61; // rax
  unsigned __int64 *v62; // rsi
  unsigned __int64 *v63; // rdx
  unsigned __int64 v64; // r8
  __int64 v65; // rdi
  __int64 v66; // rdx
  __int64 v67; // rdi
  __int64 v68; // rcx
  int v70; // edx
  int v71; // r10d
  int v72; // edx
  _BYTE *v73; // rsi
  int v74; // ecx
  __int64 *v77; // [rsp+18h] [rbp-278h]
  __int64 v79; // [rsp+20h] [rbp-270h]
  __int64 v80; // [rsp+20h] [rbp-270h]
  __int64 *v82; // [rsp+28h] [rbp-268h]
  __int64 v83; // [rsp+30h] [rbp-260h]
  __int64 v84; // [rsp+38h] [rbp-258h]
  __int64 v85; // [rsp+40h] [rbp-250h]
  __int64 v86; // [rsp+48h] [rbp-248h]
  __int64 v87; // [rsp+48h] [rbp-248h]
  __int64 v88; // [rsp+48h] [rbp-248h]
  __int64 v89; // [rsp+48h] [rbp-248h]
  __int64 *v90; // [rsp+48h] [rbp-248h]
  __int64 v91; // [rsp+48h] [rbp-248h]
  _QWORD *v92; // [rsp+50h] [rbp-240h] BYREF
  __int64 v93; // [rsp+58h] [rbp-238h]
  _QWORD v94[70]; // [rsp+60h] [rbp-230h] BYREF

  v10 = *(_QWORD *)a3;
  v84 = *(_QWORD *)(**(_QWORD **)(a3 + 32) + 56LL);
  v11 = sub_194ACF0(a6);
  v83 = (__int64)v11;
  if ( v10 )
  {
    v92 = v11;
    *v11 = v10;
    v12 = *(_BYTE **)(v10 + 16);
    if ( v12 == *(_BYTE **)(v10 + 24) )
    {
      sub_13FD960(v10 + 8, v12, &v92);
    }
    else
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = v92;
        v12 = *(_BYTE **)(v10 + 16);
      }
      *(_QWORD *)(v10 + 16) = v12 + 8;
    }
  }
  else
  {
    v92 = v11;
    v73 = *(_BYTE **)(a6 + 40);
    if ( v73 == *(_BYTE **)(a6 + 48) )
    {
      sub_13FD960(a6 + 32, v73, &v92);
    }
    else
    {
      if ( v73 )
      {
        *(_QWORD *)v73 = v11;
        v73 = *(_BYTE **)(a6 + 40);
      }
      *(_QWORD *)(a6 + 40) = v73 + 8;
    }
  }
  v86 = sub_13FC520(a3);
  v85 = sub_1AB5760(v86, a4, a5, v84, 0, 0);
  v13 = sub_1AB4240(a4, v86);
  v16 = v13[2];
  if ( v85 != v16 )
  {
    if ( v16 != 0 && v16 != -8 && v16 != -16 )
      sub_1649B30(v13);
    v13[2] = v85;
    if ( v85 != -8 && v85 != 0 && v85 != -16 )
      sub_164C220((__int64)v13);
  }
  v17 = *(unsigned int *)(a8 + 8);
  if ( (unsigned int)v17 >= *(_DWORD *)(a8 + 12) )
  {
    sub_16CD150(a8, (const void *)(a8 + 16), 0, 8, v14, v15);
    v17 = *(unsigned int *)(a8 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a8 + 8 * v17) = v85;
  ++*(_DWORD *)(a8 + 8);
  if ( v10 )
    sub_1400330(v10, v85, a6);
  sub_1AB8E10(a7, v85, a2);
  v18 = *(__int64 **)(a3 + 32);
  v77 = *(__int64 **)(a3 + 40);
  if ( v77 != v18 )
  {
    do
    {
      v87 = *v18;
      v19 = sub_1AB5760(*v18, a4, a5, v84, 0, 0);
      v20 = sub_1AB4240(a4, v87);
      v21 = v20[2];
      if ( v19 != v21 )
      {
        if ( v21 != -8 && v21 != 0 && v21 != -16 )
          sub_1649B30(v20);
        v20[2] = v19;
        if ( v19 != 0 && v19 != -8 && v19 != -16 )
          sub_164C220((__int64)v20);
      }
      sub_1400330(v83, v19, a6);
      sub_1AB8E10(a7, v19, v85);
      v24 = *(unsigned int *)(a8 + 8);
      if ( (unsigned int)v24 >= *(_DWORD *)(a8 + 12) )
      {
        sub_16CD150(a8, (const void *)(a8 + 16), 0, 8, v22, v23);
        v24 = *(unsigned int *)(a8 + 8);
      }
      ++v18;
      *(_QWORD *)(*(_QWORD *)a8 + 8 * v24) = v19;
      ++*(_DWORD *)(a8 + 8);
    }
    while ( v77 != v18 );
    v25 = *(__int64 **)(a3 + 32);
    v82 = *(__int64 **)(a3 + 40);
    while ( v82 != v25 )
    {
      v26 = *(unsigned int *)(a7 + 48);
      if ( !(_DWORD)v26 )
        goto LABEL_113;
      v27 = *(_QWORD *)(a7 + 32);
      v28 = (v26 - 1) & (((unsigned int)*v25 >> 9) ^ ((unsigned int)*v25 >> 4));
      v29 = (__int64 *)(v27 + 16LL * v28);
      v30 = *v29;
      if ( *v25 != *v29 )
      {
        v70 = 1;
        while ( v30 != -8 )
        {
          v71 = v70 + 1;
          v28 = (v26 - 1) & (v70 + v28);
          v29 = (__int64 *)(v27 + 16LL * v28);
          v30 = *v29;
          if ( *v25 == *v29 )
            goto LABEL_31;
          v70 = v71;
        }
LABEL_113:
        BUG();
      }
LABEL_31:
      v88 = *v25;
      if ( v29 == (__int64 *)(v27 + 16 * v26) )
        goto LABEL_113;
      v31 = sub_1AB4240(a4, **(_QWORD **)(v29[1] + 8))[2];
      v32 = sub_1AB4240(a4, v88);
      v33 = *(_QWORD *)(a7 + 32);
      v34 = v32[2];
      v35 = *(unsigned int *)(a7 + 48);
      if ( !(_DWORD)v35 )
        goto LABEL_112;
      v36 = v35 - 1;
      v37 = (v35 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v38 = (__int64 *)(v33 + 16LL * v37);
      v39 = *v38;
      if ( v31 == *v38 )
      {
LABEL_34:
        v40 = (__int64 *)(v33 + 16 * v35);
        if ( v38 != v40 )
        {
          v41 = v38[1];
          goto LABEL_36;
        }
      }
      else
      {
        LODWORD(v38) = 1;
        while ( v39 != -8 )
        {
          v74 = (_DWORD)v38 + 1;
          v37 = v36 & (v37 + (_DWORD)v38);
          v38 = (__int64 *)(v33 + 16LL * v37);
          v39 = *v38;
          if ( v31 == *v38 )
            goto LABEL_34;
          LODWORD(v38) = v74;
        }
        v40 = (__int64 *)(v33 + 16 * v35);
      }
      v41 = 0;
LABEL_36:
      v42 = v36 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v43 = (__int64 *)(v33 + 16LL * v42);
      v44 = *v43;
      if ( v34 != *v43 )
      {
        v72 = 1;
        while ( v44 != -8 )
        {
          LODWORD(v38) = v72 + 1;
          v42 = v36 & (v42 + v72);
          v43 = (__int64 *)(v33 + 16LL * v42);
          v44 = *v43;
          if ( v34 == *v43 )
            goto LABEL_37;
          v72 = (int)v38;
        }
LABEL_112:
        *(_BYTE *)(a7 + 72) = 0;
        BUG();
      }
LABEL_37:
      if ( v40 == v43 )
        goto LABEL_112;
      v45 = v43[1];
      *(_BYTE *)(a7 + 72) = 0;
      v46 = *(_QWORD *)(v45 + 8);
      if ( v46 != v41 )
      {
        v47 = *(char **)(v46 + 32);
        v48 = *(char **)(v46 + 24);
        v49 = (v47 - v48) >> 5;
        v50 = (v47 - v48) >> 3;
        if ( v49 > 0 )
        {
          v51 = &v48[32 * v49];
          while ( v45 != *(_QWORD *)v48 )
          {
            if ( v45 == *((_QWORD *)v48 + 1) )
            {
              v48 += 8;
              goto LABEL_46;
            }
            if ( v45 == *((_QWORD *)v48 + 2) )
            {
              v48 += 16;
              goto LABEL_46;
            }
            if ( v45 == *((_QWORD *)v48 + 3) )
            {
              v48 += 24;
              goto LABEL_46;
            }
            v48 += 32;
            if ( v48 == v51 )
            {
              v50 = (v47 - v48) >> 3;
              goto LABEL_76;
            }
          }
          goto LABEL_46;
        }
LABEL_76:
        if ( v50 != 2 )
        {
          if ( v50 != 3 )
          {
            if ( v50 != 1 )
            {
              v48 = *(char **)(v46 + 32);
LABEL_46:
              if ( v48 + 8 != v47 )
              {
                v79 = v45;
                v89 = *(_QWORD *)(v45 + 8);
                memmove(v48, v48 + 8, v47 - (v48 + 8));
                v46 = v89;
                v45 = v79;
                v47 = *(char **)(v89 + 32);
              }
              *(_QWORD *)(v46 + 32) = v47 - 8;
              *(_QWORD *)(v45 + 8) = v41;
              v92 = (_QWORD *)v45;
              v52 = *(_BYTE **)(v41 + 32);
              if ( v52 == *(_BYTE **)(v41 + 40) )
              {
                v91 = v45;
                sub_15CE310(v41 + 24, v52, &v92);
                v45 = v91;
              }
              else
              {
                if ( v52 )
                {
                  *(_QWORD *)v52 = v45;
                  v52 = *(_BYTE **)(v41 + 32);
                }
                *(_QWORD *)(v41 + 32) = v52 + 8;
              }
              if ( *(_DWORD *)(v45 + 16) != *(_DWORD *)(*(_QWORD *)(v45 + 8) + 16LL) + 1 )
              {
                v94[0] = v45;
                v92 = v94;
                v53 = v94;
                v90 = v25;
                v80 = a4;
                v93 = 0x4000000001LL;
                LODWORD(v54) = 1;
                do
                {
                  v55 = (unsigned int)v54;
                  v54 = (unsigned int)(v54 - 1);
                  v56 = v53[v55 - 1];
                  LODWORD(v93) = v54;
                  v57 = *(__int64 **)(v56 + 32);
                  v58 = *(__int64 **)(v56 + 24);
                  *(_DWORD *)(v56 + 16) = *(_DWORD *)(*(_QWORD *)(v56 + 8) + 16LL) + 1;
                  if ( v58 != v57 )
                  {
                    do
                    {
                      v59 = *v58;
                      if ( *(_DWORD *)(*v58 + 16) != *(_DWORD *)(*(_QWORD *)(*v58 + 8) + 16LL) + 1 )
                      {
                        if ( (unsigned int)v54 >= HIDWORD(v93) )
                        {
                          sub_16CD150((__int64)&v92, v94, 0, 8, (int)v38, v46);
                          v54 = (unsigned int)v93;
                        }
                        v92[v54] = v59;
                        v54 = (unsigned int)(v93 + 1);
                        LODWORD(v93) = v93 + 1;
                      }
                      ++v58;
                    }
                    while ( v57 != v58 );
                    v53 = v92;
                  }
                }
                while ( (_DWORD)v54 );
                v25 = v90;
                a4 = v80;
                if ( v53 != v94 )
                  _libc_free((unsigned __int64)v53);
              }
              goto LABEL_64;
            }
LABEL_88:
            if ( v45 != *(_QWORD *)v48 )
              v48 = *(char **)(v46 + 32);
            goto LABEL_46;
          }
          if ( v45 == *(_QWORD *)v48 )
            goto LABEL_46;
          v48 += 8;
        }
        if ( v45 == *(_QWORD *)v48 )
          goto LABEL_46;
        v48 += 8;
        goto LABEL_88;
      }
LABEL_64:
      ++v25;
    }
  }
  v60 = v84 + 72;
  v61 = (unsigned __int64 *)(a1 + 24);
  if ( !v85 )
    BUG();
  v62 = (unsigned __int64 *)(v85 + 24);
  v63 = *(unsigned __int64 **)(v85 + 32);
  if ( v61 != (unsigned __int64 *)(v85 + 24) && v61 != v63 && v62 != v63 )
  {
    v64 = *v63 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)((*(_QWORD *)(v85 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 8) = v63;
    *v63 = *v63 & 7 | *(_QWORD *)(v85 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    v65 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(v64 + 8) = v61;
    v65 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v85 + 24) = v65 | *(_QWORD *)(v85 + 24) & 7LL;
    *(_QWORD *)(v65 + 8) = v62;
    *(_QWORD *)(a1 + 24) = v64 | *(_QWORD *)(a1 + 24) & 7LL;
  }
  v66 = **(_QWORD **)(v83 + 32);
  if ( v60 != v66 + 24 && (unsigned __int64 *)v60 != v61 )
  {
    v67 = *(_QWORD *)(v84 + 72);
    *(_QWORD *)((*(_QWORD *)(v66 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 8) = v60;
    v67 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v84 + 72) = *(_QWORD *)(v84 + 72) & 7LL | *(_QWORD *)(v66 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    v68 = *(_QWORD *)(a1 + 24);
    *(_QWORD *)(v67 + 8) = v61;
    v68 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v66 + 24) = v68 | *(_QWORD *)(v66 + 24) & 7LL;
    *(_QWORD *)(v68 + 8) = v66 + 24;
    *(_QWORD *)(a1 + 24) = v67 | *(_QWORD *)(a1 + 24) & 7LL;
  }
  return v83;
}
