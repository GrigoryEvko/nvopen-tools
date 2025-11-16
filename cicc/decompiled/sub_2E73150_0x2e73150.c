// Function: sub_2E73150
// Address: 0x2e73150
//
void __fastcall sub_2E73150(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rsi
  unsigned int v11; // r15d
  unsigned __int64 *v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r8
  __int64 v17; // rdx
  unsigned __int64 v18; // rsi
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rcx
  unsigned __int64 v27; // rdx
  __int64 *v28; // rbx
  char *v29; // r14
  __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 v32; // rax
  unsigned int v33; // r13d
  __int64 *v34; // rdi
  int v35; // esi
  unsigned int v36; // edx
  __int64 v37; // rax
  unsigned int v38; // esi
  unsigned int v39; // edx
  unsigned int v40; // edi
  bool v41; // cf
  __int64 v42; // r13
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  __int64 v47; // r13
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  unsigned __int64 v50; // rsi
  __int64 v51; // rax
  __int64 v52; // rcx
  __int64 v53; // rdx
  unsigned __int64 v54; // rdi
  __int64 *v55; // rcx
  int v56; // r11d
  __int64 *v57; // r10
  __int64 *v58; // r12
  __int64 *v59; // rbx
  __int64 v60; // rdi
  __int64 *v61; // rax
  __int64 v62; // r10
  __int64 v63; // r11
  __int64 i; // r8
  __int64 v65; // rax
  unsigned __int64 v66; // rdi
  __int64 v67; // rsi
  __int64 v68; // rcx
  unsigned __int64 v69; // rdi
  _QWORD *v70; // rax
  __int64 v71; // rax
  __int64 v72; // rcx
  _QWORD *v73; // r8
  __int64 v74; // [rsp+8h] [rbp-298h]
  unsigned int v76; // [rsp+64h] [rbp-23Ch]
  __int64 *v77; // [rsp+68h] [rbp-238h] BYREF
  __int64 v78; // [rsp+70h] [rbp-230h] BYREF
  __int64 *v79; // [rsp+78h] [rbp-228h] BYREF
  _BYTE *v80; // [rsp+80h] [rbp-220h] BYREF
  __int64 v81; // [rsp+88h] [rbp-218h]
  _BYTE v82[64]; // [rsp+90h] [rbp-210h] BYREF
  char *v83; // [rsp+D0h] [rbp-1D0h] BYREF
  int v84; // [rsp+D8h] [rbp-1C8h]
  char v85; // [rsp+E0h] [rbp-1C0h] BYREF
  _BYTE *v86; // [rsp+120h] [rbp-180h] BYREF
  __int64 v87; // [rsp+128h] [rbp-178h]
  _BYTE v88[72]; // [rsp+130h] [rbp-170h] BYREF
  __int64 v89; // [rsp+178h] [rbp-128h] BYREF
  __int64 v90; // [rsp+180h] [rbp-120h]
  __int64 *v91; // [rsp+188h] [rbp-118h] BYREF
  unsigned int v92; // [rsp+190h] [rbp-110h]
  __int64 *v93; // [rsp+1C8h] [rbp-D8h] BYREF
  __int64 v94; // [rsp+1D0h] [rbp-D0h]
  _BYTE v95[64]; // [rsp+1D8h] [rbp-C8h] BYREF
  _BYTE *v96; // [rsp+218h] [rbp-88h] BYREF
  __int64 v97; // [rsp+220h] [rbp-80h]
  _BYTE v98[120]; // [rsp+228h] [rbp-78h] BYREF

  v77 = a4;
  if ( a3 && *a4 && (v7 = sub_2E6D000(a1, a3, *a4)) != 0 )
  {
    v8 = (unsigned int)(*(_DWORD *)(v7 + 24) + 1);
    v9 = *(_DWORD *)(v7 + 24) + 1;
  }
  else
  {
    v8 = 0;
    v9 = 0;
  }
  if ( v9 >= *(_DWORD *)(a1 + 32) )
LABEL_107:
    BUG();
  v10 = (__int64)v77;
  v74 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v8);
  v11 = *(_DWORD *)(v74 + 16) + 1;
  if ( v11 < *((_DWORD *)v77 + 4) )
  {
    v89 = 0;
    v86 = v88;
    v87 = 0x800000000LL;
    v12 = (unsigned __int64 *)&v91;
    v90 = 1;
    do
      *v12++ = -4096;
    while ( v12 != (unsigned __int64 *)&v93 );
    v93 = (__int64 *)v95;
    v96 = v98;
    v94 = 0x800000000LL;
    v97 = 0x800000000LL;
    v81 = 0x800000000LL;
    v80 = v82;
    sub_2E6D8E0((__int64)&v86, v10, (__int64)&v93, (__int64)a4, a5, a6);
    v13 = (unsigned __int64)v86;
    v14 = 8LL * (unsigned int)v87;
    v15 = *(_QWORD *)&v86[v14 - 8];
    v16 = (v14 >> 3) - 1;
    v17 = ((v14 >> 3) - 2) / 2;
    if ( v16 > 0 )
    {
      while ( 1 )
      {
        v18 = v13 + 8 * v17;
        v73 = (_QWORD *)(v13 + 8 * v16);
        if ( *(_DWORD *)(*(_QWORD *)v18 + 16LL) >= *(_DWORD *)(v15 + 16) )
          break;
        *v73 = *(_QWORD *)v18;
        v16 = v17;
        if ( v17 <= 0 )
        {
          v73 = (_QWORD *)(v13 + 8 * v17);
          break;
        }
        v17 = (v17 - 1) / 2;
      }
    }
    else
    {
      v73 = &v86[v14 - 8];
    }
    *v73 = v15;
    sub_2E73000((__int64)&v83, (__int64)&v89, (__int64 *)&v77);
    v21 = (unsigned int)v87;
    if ( !(_DWORD)v87 )
      goto LABEL_68;
    while ( 1 )
    {
      v22 = (unsigned __int64)v86;
      v23 = 8 * v21;
      v24 = *(_QWORD *)v86;
      if ( v21 != 1 )
        break;
LABEL_18:
      v25 = (unsigned int)v94;
      v26 = HIDWORD(v94);
      LODWORD(v87) = v87 - 1;
      v27 = (unsigned int)v94 + 1LL;
      if ( v27 > HIDWORD(v94) )
      {
        sub_C8D5F0((__int64)&v93, v95, v27, 8u, v19, v20);
        v25 = (unsigned int)v94;
      }
      v93[v25] = v24;
      LODWORD(v94) = v94 + 1;
      v76 = *(_DWORD *)(v24 + 16);
      while ( 1 )
      {
        sub_2E6EC80(&v83, *(_QWORD *)v24, a2, v26, v19);
        v28 = (__int64 *)v83;
        v29 = &v83[8 * v84];
        if ( v83 != v29 )
        {
          while ( 1 )
          {
            v37 = *v28;
            if ( *v28 )
            {
              v30 = (unsigned int)(*(_DWORD *)(v37 + 24) + 1);
              v31 = *(_DWORD *)(v37 + 24) + 1;
            }
            else
            {
              v30 = 0;
              v31 = 0;
            }
            if ( v31 >= *(_DWORD *)(a1 + 32) )
            {
              v78 = 0;
              goto LABEL_107;
            }
            v32 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v30);
            v33 = *(_DWORD *)(v32 + 16);
            v78 = v32;
            if ( v11 < v33 )
            {
              if ( (v90 & 1) != 0 )
              {
                v34 = (__int64 *)&v91;
                v35 = 7;
              }
              else
              {
                v38 = v92;
                v34 = v91;
                if ( !v92 )
                {
                  v39 = v90;
                  ++v89;
                  v79 = 0;
                  v40 = ((unsigned int)v90 >> 1) + 1;
                  goto LABEL_40;
                }
                v35 = v92 - 1;
              }
              v36 = v35 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
              v20 = (__int64)&v34[v36];
              v19 = *(_QWORD *)v20;
              if ( v32 != *(_QWORD *)v20 )
                break;
            }
LABEL_29:
            if ( v29 == (char *)++v28 )
            {
              v29 = v83;
              goto LABEL_35;
            }
          }
          v56 = 1;
          v57 = 0;
          while ( v19 != -4096 )
          {
            if ( v57 || v19 != -8192 )
              v20 = (__int64)v57;
            v36 = v35 & (v56 + v36);
            v19 = v34[v36];
            if ( v32 == v19 )
              goto LABEL_29;
            ++v56;
            v57 = (__int64 *)v20;
            v20 = (__int64)&v34[v36];
          }
          v39 = v90;
          if ( !v57 )
            v57 = (__int64 *)v20;
          ++v89;
          v79 = v57;
          v40 = ((unsigned int)v90 >> 1) + 1;
          if ( (v90 & 1) != 0 )
          {
            v19 = 24;
            v38 = 8;
            if ( 4 * v40 >= 0x18 )
              goto LABEL_65;
LABEL_41:
            if ( v38 - HIDWORD(v90) - v40 > v38 >> 3 )
            {
LABEL_42:
              LODWORD(v90) = (2 * (v39 >> 1) + 2) | v39 & 1;
              if ( *v79 != -4096 )
                --HIDWORD(v90);
              v41 = v76 < v33;
              *v79 = v32;
              v42 = v78;
              if ( v41 )
              {
                v43 = (unsigned int)v81;
                v44 = (unsigned int)v81 + 1LL;
                if ( v44 > HIDWORD(v81) )
                {
                  sub_C8D5F0((__int64)&v80, v82, v44, 8u, v19, v20);
                  v43 = (unsigned int)v81;
                }
                *(_QWORD *)&v80[8 * v43] = v42;
                v45 = (unsigned int)v97;
                LODWORD(v81) = v81 + 1;
                v46 = (unsigned int)v97 + 1LL;
                v47 = v78;
                if ( v46 > HIDWORD(v97) )
                {
                  sub_C8D5F0((__int64)&v96, v98, v46, 8u, v19, v20);
                  v45 = (unsigned int)v97;
                }
                *(_QWORD *)&v96[8 * v45] = v47;
                LODWORD(v97) = v97 + 1;
              }
              else
              {
                v48 = (unsigned int)v87;
                v49 = (unsigned int)v87 + 1LL;
                if ( v49 > HIDWORD(v87) )
                {
                  sub_C8D5F0((__int64)&v86, v88, v49, 8u, v19, v20);
                  v48 = (unsigned int)v87;
                }
                *(_QWORD *)&v86[8 * v48] = v42;
                v50 = (unsigned __int64)v86;
                LODWORD(v87) = v87 + 1;
                v51 = 8LL * (unsigned int)v87;
                v19 = *(_QWORD *)&v86[v51 - 8];
                v52 = (v51 >> 3) - 1;
                v53 = ((v51 >> 3) - 2) / 2;
                if ( v52 > 0 )
                {
                  while ( 1 )
                  {
                    v54 = v50 + 8 * v53;
                    v55 = (__int64 *)(v50 + 8 * v52);
                    if ( *(_DWORD *)(*(_QWORD *)v54 + 16LL) >= *(_DWORD *)(v19 + 16) )
                    {
                      *v55 = v19;
                      goto LABEL_29;
                    }
                    *v55 = *(_QWORD *)v54;
                    v52 = v53;
                    if ( v53 <= 0 )
                      break;
                    v53 = (v53 - 1) / 2;
                  }
                  *(_QWORD *)v54 = v19;
                }
                else
                {
                  *(_QWORD *)&v86[v51 - 8] = v19;
                }
              }
              goto LABEL_29;
            }
          }
          else
          {
            v38 = v92;
LABEL_40:
            v19 = 3 * v38;
            if ( 4 * v40 < (unsigned int)v19 )
              goto LABEL_41;
LABEL_65:
            v38 *= 2;
          }
          sub_2E72D00((__int64)&v89, v38);
          sub_2E6EF00((__int64)&v89, &v78, &v79);
          v32 = v78;
          v39 = v90;
          goto LABEL_42;
        }
LABEL_35:
        if ( v29 != &v85 )
          _libc_free((unsigned __int64)v29);
        if ( !(_DWORD)v81 )
          break;
        v26 = (unsigned int)v81;
        v24 = *(_QWORD *)&v80[8 * (unsigned int)v81 - 8];
        LODWORD(v81) = v81 - 1;
      }
      v21 = (unsigned int)v87;
      if ( !(_DWORD)v87 )
      {
LABEL_68:
        v58 = v93;
        v59 = &v93[(unsigned int)v94];
        if ( v93 != v59 )
        {
          do
          {
            v60 = *v58++;
            sub_2E6CC90(v60, v74);
          }
          while ( v59 != v58 );
        }
        if ( v80 != v82 )
          _libc_free((unsigned __int64)v80);
        if ( v96 != v98 )
          _libc_free((unsigned __int64)v96);
        if ( v93 != (__int64 *)v95 )
          _libc_free((unsigned __int64)v93);
        if ( (v90 & 1) == 0 )
          sub_C7D6A0((__int64)v91, 8LL * v92, 8);
        if ( v86 != v88 )
          _libc_free((unsigned __int64)v86);
        return;
      }
    }
    v61 = (__int64 *)&v86[v23 - 8];
    v62 = *v61;
    *v61 = v24;
    v63 = (v23 - 8) >> 3;
    v20 = (v63 - 1) / 2;
    if ( v23 - 8 <= 16 )
    {
      v70 = (_QWORD *)v22;
      if ( (v63 & 1) != 0 || (unsigned __int64)(v63 - 1) > 2 )
      {
LABEL_92:
        *v70 = v62;
        goto LABEL_18;
      }
      v66 = v22;
      v65 = 0;
    }
    else
    {
      for ( i = 0; ; i = v65 )
      {
        v65 = 2 * (i + 1);
        v66 = v22 + 16 * (i + 1);
        v67 = *(_QWORD *)v66;
        if ( *(_DWORD *)(*(_QWORD *)v66 + 16LL) < *(_DWORD *)(*(_QWORD *)(v66 - 8) + 16LL) )
        {
          --v65;
          v66 = v22 + 8 * v65;
          v67 = *(_QWORD *)v66;
        }
        *(_QWORD *)(v22 + 8 * i) = v67;
        if ( v65 >= v20 )
          break;
      }
      if ( (v63 & 1) != 0 )
      {
LABEL_101:
        v19 = v65;
        v68 = (v65 - 1) >> 1;
LABEL_91:
        while ( 1 )
        {
          v69 = v22 + 8 * v68;
          v20 = *(unsigned int *)(v62 + 16);
          v70 = (_QWORD *)(v22 + 8 * v19);
          if ( *(_DWORD *)(*(_QWORD *)v69 + 16LL) >= (unsigned int)v20 )
            goto LABEL_92;
          *v70 = *(_QWORD *)v69;
          v19 = v68;
          if ( !v68 )
          {
            *(_QWORD *)v69 = v62;
            goto LABEL_18;
          }
          v68 = (v68 - 1) / 2;
        }
      }
      v19 = v65;
      v68 = (v65 - 1) >> 1;
      if ( v65 != (v63 - 2) / 2 )
        goto LABEL_91;
    }
    v71 = 2 * v65 + 2;
    v72 = *(_QWORD *)(v22 + 8 * v71 - 8);
    v65 = v71 - 1;
    *(_QWORD *)v66 = v72;
    goto LABEL_101;
  }
}
