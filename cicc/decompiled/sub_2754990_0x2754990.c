// Function: sub_2754990
// Address: 0x2754990
//
void __fastcall sub_2754990(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  _QWORD *v5; // rdi
  unsigned int v6; // eax
  __int64 v7; // r12
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  int v11; // ecx
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rsi
  unsigned int v21; // ecx
  __int64 *v22; // rdx
  __int64 v23; // r9
  __int64 v24; // rcx
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rsi
  unsigned int v28; // ecx
  __int64 *v29; // rdx
  __int64 v30; // r9
  __int64 v31; // rcx
  _QWORD *v32; // rsi
  __int64 v33; // rax
  __int64 v34; // r8
  __int64 v35; // r13
  __int64 i; // r15
  __int64 ***v37; // r14
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rax
  _QWORD *v48; // rbx
  _QWORD *v49; // r12
  __int64 v50; // rax
  _QWORD *v51; // rbx
  _QWORD *v52; // r12
  __int64 v53; // rax
  int v54; // eax
  _QWORD *v55; // rax
  _QWORD *v56; // rax
  _BYTE *v57; // rax
  __int64 v58; // rdi
  int v59; // edx
  int v60; // r10d
  int v61; // edx
  int v62; // r10d
  unsigned __int8 *v63; // rax
  int v64; // esi
  __int64 v65; // rdi
  unsigned __int8 *v66; // rdx
  int v67; // esi
  __int64 *v68; // rax
  int v69; // esi
  __int64 v70; // rdi
  int v71; // esi
  __int64 *v72; // rax
  int v73; // eax
  int v74; // eax
  char v76; // [rsp+3Fh] [rbp-441h]
  _QWORD *v77; // [rsp+40h] [rbp-440h] BYREF
  __int64 v78; // [rsp+48h] [rbp-438h]
  _QWORD v79[32]; // [rsp+50h] [rbp-430h] BYREF
  __int64 v80; // [rsp+150h] [rbp-330h] BYREF
  _BYTE *v81; // [rsp+158h] [rbp-328h]
  __int64 v82; // [rsp+160h] [rbp-320h]
  _BYTE v83[384]; // [rsp+168h] [rbp-318h] BYREF
  __int64 v84; // [rsp+2E8h] [rbp-198h]
  char *v85; // [rsp+2F0h] [rbp-190h]
  __int64 v86; // [rsp+2F8h] [rbp-188h]
  int v87; // [rsp+300h] [rbp-180h]
  char v88; // [rsp+304h] [rbp-17Ch]
  char v89; // [rsp+308h] [rbp-178h] BYREF
  _BYTE *v90; // [rsp+348h] [rbp-138h]
  __int64 v91; // [rsp+350h] [rbp-130h]
  _BYTE v92[200]; // [rsp+358h] [rbp-128h] BYREF
  int v93; // [rsp+420h] [rbp-60h] BYREF
  _QWORD *v94; // [rsp+428h] [rbp-58h]
  int *v95; // [rsp+430h] [rbp-50h]
  int *v96; // [rsp+438h] [rbp-48h]
  __int64 v97; // [rsp+440h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 784);
  v79[0] = a2;
  v5 = v79;
  v80 = v4;
  v81 = v83;
  v82 = 0x1000000000LL;
  v85 = &v89;
  v90 = v92;
  v91 = 0x800000000LL;
  v95 = &v93;
  v96 = &v93;
  v78 = 0x2000000001LL;
  v6 = 1;
  v84 = 0;
  v86 = 8;
  v87 = 0;
  v88 = 1;
  v93 = 0;
  v94 = 0;
  v97 = 0;
  v77 = v79;
  do
  {
    v7 = v5[v6 - 1];
    LODWORD(v78) = v6 - 1;
    sub_F54ED0((unsigned __int8 *)v7);
    sub_11C4E30((unsigned __int8 *)v7, 0, 0);
    v10 = *(_QWORD *)(a1 + 784);
    v11 = *(_DWORD *)(v10 + 56);
    v12 = *(_QWORD *)(v10 + 40);
    if ( v11 )
    {
      v13 = (unsigned int)(v11 - 1);
      v14 = (unsigned int)v13 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v15 = (__int64 *)(v12 + 16 * v14);
      v16 = *v15;
      if ( v7 == *v15 )
      {
LABEL_4:
        v17 = v15[1];
        if ( v17 )
        {
          if ( *(_BYTE *)v17 != 27 )
          {
            v76 = 0;
            goto LABEL_7;
          }
          if ( !*(_BYTE *)(a1 + 1396) )
            goto LABEL_81;
          v55 = *(_QWORD **)(a1 + 1376);
          v13 = *(unsigned int *)(a1 + 1388);
          v14 = (__int64)&v55[v13];
          if ( v55 == (_QWORD *)v14 )
          {
LABEL_80:
            if ( (unsigned int)v13 >= *(_DWORD *)(a1 + 1384) )
            {
LABEL_81:
              sub_C8CC70(a1 + 1368, v17, v14, v13, v8, v9);
              goto LABEL_70;
            }
            v13 = (unsigned int)(v13 + 1);
            *(_DWORD *)(a1 + 1388) = v13;
            *(_QWORD *)v14 = v17;
            ++*(_QWORD *)(a1 + 1368);
          }
          else
          {
            while ( v17 != *v55 )
            {
              if ( (_QWORD *)v14 == ++v55 )
                goto LABEL_80;
            }
          }
LABEL_70:
          if ( a3 )
          {
            if ( !*(_BYTE *)(a3 + 28) )
              goto LABEL_87;
            v56 = *(_QWORD **)(a3 + 8);
            v13 = *(unsigned int *)(a3 + 20);
            v14 = (__int64)&v56[v13];
            if ( v56 != (_QWORD *)v14 )
            {
              while ( v17 != *v56 )
              {
                if ( (_QWORD *)v14 == ++v56 )
                  goto LABEL_86;
              }
              goto LABEL_76;
            }
LABEL_86:
            if ( (unsigned int)v13 < *(_DWORD *)(a3 + 16) )
            {
              v13 = (unsigned int)(v13 + 1);
              *(_DWORD *)(a3 + 20) = v13;
              *(_QWORD *)v14 = v17;
              ++*(_QWORD *)a3;
            }
            else
            {
LABEL_87:
              sub_C8CC70(a3, v17, v14, v13, v8, v9);
            }
          }
LABEL_76:
          v57 = *(_BYTE **)(v17 + 72);
          if ( *v57 == 62 )
          {
            v58 = *((_QWORD *)v57 - 8);
            if ( *(_BYTE *)(*(_QWORD *)(v58 + 8) + 8LL) == 14 )
            {
              v63 = sub_98ACB0((unsigned __int8 *)v58, 6u);
              v64 = *(_DWORD *)(a1 + 1456);
              v65 = *(_QWORD *)(a1 + 1440);
              v66 = v63;
              if ( v64 )
              {
                v67 = v64 - 1;
                v13 = v67 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
                v68 = (__int64 *)(v65 + 16 * v13);
                v9 = *v68;
                if ( v66 == (unsigned __int8 *)*v68 )
                {
LABEL_95:
                  *v68 = -8192;
                  --*(_DWORD *)(a1 + 1448);
                  ++*(_DWORD *)(a1 + 1452);
                  *(_BYTE *)(a1 + 1737) = 1;
                }
                else
                {
                  v73 = 1;
                  while ( v9 != -4096 )
                  {
                    v8 = (unsigned int)(v73 + 1);
                    v13 = v67 & (unsigned int)(v73 + v13);
                    v68 = (__int64 *)(v65 + 16LL * (unsigned int)v13);
                    v9 = *v68;
                    if ( v66 == (unsigned __int8 *)*v68 )
                      goto LABEL_95;
                    v73 = v8;
                  }
                }
              }
              v69 = *(_DWORD *)(a1 + 1488);
              v70 = *(_QWORD *)(a1 + 1472);
              if ( v69 )
              {
                v71 = v69 - 1;
                v13 = v71 & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
                v72 = (__int64 *)(v70 + 16 * v13);
                v9 = *v72;
                if ( v66 == (unsigned __int8 *)*v72 )
                {
LABEL_98:
                  *v72 = -8192;
                  v76 = 1;
                  --*(_DWORD *)(a1 + 1480);
                  ++*(_DWORD *)(a1 + 1484);
                  goto LABEL_7;
                }
                v74 = 1;
                while ( v9 != -4096 )
                {
                  v8 = (unsigned int)(v74 + 1);
                  v13 = v71 & (unsigned int)(v74 + v13);
                  v72 = (__int64 *)(v70 + 16LL * (unsigned int)v13);
                  v9 = *v72;
                  if ( v66 == (unsigned __int8 *)*v72 )
                    goto LABEL_98;
                  v74 = v8;
                }
              }
            }
          }
          v76 = 1;
LABEL_7:
          sub_D6E4B0(&v80, v17, 0, v13, v8, v9);
          goto LABEL_8;
        }
      }
      else
      {
        v54 = 1;
        while ( v16 != -4096 )
        {
          v8 = (unsigned int)(v54 + 1);
          v14 = (unsigned int)v13 & (v54 + (_DWORD)v14);
          v15 = (__int64 *)(v12 + 16LL * (unsigned int)v14);
          v16 = *v15;
          if ( v7 == *v15 )
            goto LABEL_4;
          v54 = v8;
        }
      }
    }
    v76 = 0;
LABEL_8:
    v18 = *(unsigned int *)(a1 + 1712);
    v19 = *(_QWORD *)(v7 + 40);
    v20 = *(_QWORD *)(a1 + 1696);
    if ( (_DWORD)v18 )
    {
      v21 = (v18 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v22 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v22;
      if ( v19 == *v22 )
      {
LABEL_10:
        if ( v22 != (__int64 *)(v20 + 16 * v18) )
        {
          v24 = *(_QWORD *)(a1 + 1720);
          v25 = v24 + 56LL * *((unsigned int *)v22 + 2);
          if ( v25 != v24 + 56LL * *(unsigned int *)(a1 + 1728) )
          {
            v26 = *(unsigned int *)(v25 + 32);
            v27 = *(_QWORD *)(v25 + 16);
            if ( (_DWORD)v26 )
            {
              v28 = (v26 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
              v29 = (__int64 *)(v27 + 16LL * v28);
              v30 = *v29;
              if ( v7 == *v29 )
              {
LABEL_14:
                if ( v29 != (__int64 *)(v27 + 16 * v26) )
                {
                  v31 = *(_QWORD *)(v25 + 40);
                  v32 = (_QWORD *)(v31 + 56LL * *((unsigned int *)v29 + 2));
                  if ( v32 != (_QWORD *)(v31 + 56LL * *(unsigned int *)(v25 + 48)) )
                    sub_27546E0(v25 + 8, v32);
                }
              }
              else
              {
                v61 = 1;
                while ( v30 != -4096 )
                {
                  v62 = v61 + 1;
                  v28 = (v26 - 1) & (v61 + v28);
                  v29 = (__int64 *)(v27 + 16LL * v28);
                  v30 = *v29;
                  if ( v7 == *v29 )
                    goto LABEL_14;
                  v61 = v62;
                }
              }
            }
          }
        }
      }
      else
      {
        v59 = 1;
        while ( v23 != -4096 )
        {
          v60 = v59 + 1;
          v21 = (v18 - 1) & (v59 + v21);
          v22 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v22;
          if ( v19 == *v22 )
            goto LABEL_10;
          v59 = v60;
        }
      }
    }
    v33 = 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
    {
      v34 = *(_QWORD *)(v7 - 8);
      v35 = v34 + v33;
    }
    else
    {
      v35 = v7;
      v34 = v7 - v33;
    }
    for ( i = v34; v35 != i; LODWORD(v78) = v78 + 1 )
    {
      while ( 1 )
      {
        v37 = *(__int64 ****)i;
        if ( **(_BYTE **)i > 0x1Cu )
        {
          v38 = sub_ACADE0(v37[1]);
          if ( *(_QWORD *)i )
          {
            v39 = *(_QWORD *)(i + 8);
            **(_QWORD **)(i + 16) = v39;
            if ( v39 )
              *(_QWORD *)(v39 + 16) = *(_QWORD *)(i + 16);
          }
          *(_QWORD *)i = v38;
          if ( v38 )
          {
            v40 = *(_QWORD *)(v38 + 16);
            *(_QWORD *)(i + 8) = v40;
            if ( v40 )
              *(_QWORD *)(v40 + 16) = i + 8;
            *(_QWORD *)(i + 16) = v38 + 16;
            *(_QWORD *)(v38 + 16) = i;
          }
          if ( sub_F50EE0((unsigned __int8 *)v37, *(__int64 **)(a1 + 808)) )
            break;
        }
        i += 32;
        if ( v35 == i )
          goto LABEL_34;
      }
      v43 = (unsigned int)v78;
      v44 = (unsigned int)v78 + 1LL;
      if ( v44 > HIDWORD(v78) )
      {
        sub_C8D5F0((__int64)&v77, v79, v44, 8u, v41, v42);
        v43 = (unsigned int)v78;
      }
      i += 32;
      v77[v43] = v37;
    }
LABEL_34:
    sub_D03960(a1 + 16, v7);
    if ( v76 && *(_BYTE *)(*(_QWORD *)(v7 + 8) + 8LL) == 7 )
    {
      sub_B43D60((_QWORD *)v7);
    }
    else
    {
      v47 = *(unsigned int *)(a1 + 1752);
      if ( v47 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1756) )
      {
        sub_C8D5F0(a1 + 1744, (const void *)(a1 + 1760), v47 + 1, 8u, v45, v46);
        v47 = *(unsigned int *)(a1 + 1752);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 1744) + 8 * v47) = v7;
      ++*(_DWORD *)(a1 + 1752);
    }
    v6 = v78;
    v5 = v77;
  }
  while ( (_DWORD)v78 );
  if ( v77 != v79 )
    _libc_free((unsigned __int64)v77);
  sub_2753F50(v94);
  v48 = v90;
  v49 = &v90[24 * (unsigned int)v91];
  if ( v90 != (_BYTE *)v49 )
  {
    do
    {
      v50 = *(v49 - 1);
      v49 -= 3;
      if ( v50 != -4096 && v50 != 0 && v50 != -8192 )
        sub_BD60C0(v49);
    }
    while ( v48 != v49 );
    v49 = v90;
  }
  if ( v49 != (_QWORD *)v92 )
    _libc_free((unsigned __int64)v49);
  if ( !v88 )
    _libc_free((unsigned __int64)v85);
  v51 = v81;
  v52 = &v81[24 * (unsigned int)v82];
  if ( v81 != (_BYTE *)v52 )
  {
    do
    {
      v53 = *(v52 - 1);
      v52 -= 3;
      if ( v53 != -4096 && v53 != 0 && v53 != -8192 )
        sub_BD60C0(v52);
    }
    while ( v51 != v52 );
    v52 = v81;
  }
  if ( v52 != (_QWORD *)v83 )
    _libc_free((unsigned __int64)v52);
}
