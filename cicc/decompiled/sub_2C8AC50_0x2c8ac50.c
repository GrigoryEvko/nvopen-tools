// Function: sub_2C8AC50
// Address: 0x2c8ac50
//
__int64 __fastcall sub_2C8AC50(__int64 *a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 *v5; // rax
  __int64 v6; // r11
  __int64 v7; // rbx
  __int64 v9; // r9
  __int64 v10; // r10
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rcx
  unsigned int v14; // edx
  __int64 *v15; // r13
  __int64 v16; // rsi
  int v17; // eax
  unsigned int v18; // edx
  _DWORD *v19; // rcx
  int v20; // esi
  _DWORD *v21; // r8
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rdi
  int v27; // r8d
  __int64 v28; // rax
  unsigned int j; // edx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 *v32; // rax
  __int64 v33; // r15
  unsigned int v34; // r11d
  __int64 v35; // r13
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // r8
  unsigned int v41; // edi
  __int64 *v42; // rbx
  __int64 v43; // r10
  int v44; // edx
  int v45; // r10d
  _DWORD *v46; // r12
  unsigned int v47; // esi
  _DWORD *v48; // r8
  int v49; // edi
  int v50; // eax
  unsigned int v51; // r12d
  int v53; // ebx
  int v54; // r13d
  int v55; // r8d
  unsigned int i; // edx
  __int64 v57; // rax
  __int64 v58; // rax
  int v59; // r9d
  unsigned int v60; // edx
  int v61; // esi
  int v62; // r8d
  _DWORD *v63; // rcx
  int v64; // edi
  int v65; // r9d
  unsigned int v66; // edx
  int v67; // esi
  int v68; // edi
  unsigned int v69; // edx
  int v70; // esi
  _DWORD *v71; // rcx
  int v72; // edi
  int v73; // esi
  unsigned int v74; // edx
  const void *v75; // [rsp+0h] [rbp-80h]
  __int64 v76; // [rsp+10h] [rbp-70h]
  __int64 v77; // [rsp+10h] [rbp-70h]
  int v78; // [rsp+18h] [rbp-68h]
  int v79; // [rsp+18h] [rbp-68h]
  __int64 v80; // [rsp+18h] [rbp-68h]
  int v82; // [rsp+20h] [rbp-60h]
  int v84; // [rsp+28h] [rbp-58h]
  __int64 v85; // [rsp+30h] [rbp-50h] BYREF
  __int64 v86; // [rsp+38h] [rbp-48h]
  __int64 v87; // [rsp+40h] [rbp-40h]
  __int64 v88; // [rsp+48h] [rbp-38h]

  v5 = (__int64 *)*a1;
  v87 = 0;
  v88 = 0;
  v85 = 0;
  v86 = 0;
  v6 = *v5;
  v7 = *(_QWORD *)(*v5 + 16);
  if ( v7 )
  {
    v9 = 0;
    v10 = 0;
    v75 = (const void *)(a4 + 16);
    while ( 1 )
    {
      v11 = *(_QWORD *)(v7 + 24);
      if ( *(_BYTE *)v11 > 0x1Cu && *(_QWORD *)(v11 + 40) == *(_QWORD *)(v6 + 40) )
      {
        v12 = *(unsigned int *)(a2 + 24);
        v13 = *(_QWORD *)(a2 + 8);
        if ( (_DWORD)v12 )
        {
          v14 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v15 = (__int64 *)(v13 + 16LL * v14);
          v16 = *v15;
          if ( v11 != *v15 )
          {
            v54 = 1;
            while ( v16 != -4096 )
            {
              v55 = v54 + 1;
              v14 = (v12 - 1) & (v54 + v14);
              v15 = (__int64 *)(v13 + 16LL * v14);
              v16 = *v15;
              if ( v11 == *v15 )
                goto LABEL_8;
              v54 = v55;
            }
            goto LABEL_3;
          }
LABEL_8:
          if ( v15 != (__int64 *)(v13 + 16 * v12) )
          {
            if ( !(_DWORD)v9 )
            {
              ++v85;
              goto LABEL_93;
            }
            v17 = *((_DWORD *)v15 + 2);
            v18 = (v9 - 1) & (37 * v17);
            v19 = (_DWORD *)(v10 + 4LL * v18);
            v20 = *v19;
            if ( v17 != *v19 )
            {
              v78 = 1;
              v21 = 0;
              while ( v20 != -1 )
              {
                if ( v20 == -2 && !v21 )
                  v21 = v19;
                v18 = (v9 - 1) & (v78 + v18);
                v19 = (_DWORD *)(v10 + 4LL * v18);
                v20 = *v19;
                if ( v17 == *v19 )
                  goto LABEL_3;
                ++v78;
              }
              if ( !v21 )
                v21 = v19;
              ++v85;
              v22 = v87 + 1;
              if ( 4 * ((int)v87 + 1) < (unsigned int)(3 * v9) )
              {
                if ( (int)v9 - (v22 + HIDWORD(v87)) <= (unsigned int)v9 >> 3 )
                {
                  v77 = v6;
                  sub_A08C50((__int64)&v85, v9);
                  if ( !(_DWORD)v88 )
                  {
LABEL_129:
                    LODWORD(v87) = v87 + 1;
                    BUG();
                  }
                  v72 = *((_DWORD *)v15 + 2);
                  v73 = 1;
                  v6 = v77;
                  v74 = (v88 - 1) & (37 * v72);
                  v71 = 0;
                  v21 = (_DWORD *)(v86 + 4LL * v74);
                  v9 = (unsigned int)*v21;
                  v22 = v87 + 1;
                  if ( v72 != (_DWORD)v9 )
                  {
                    while ( (_DWORD)v9 != -1 )
                    {
                      if ( (_DWORD)v9 == -2 && !v71 )
                        v71 = v21;
                      v74 = (v88 - 1) & (v73 + v74);
                      v21 = (_DWORD *)(v86 + 4LL * v74);
                      v9 = (unsigned int)*v21;
                      if ( v72 == (_DWORD)v9 )
                        goto LABEL_17;
                      ++v73;
                    }
                    goto LABEL_105;
                  }
                }
                goto LABEL_17;
              }
LABEL_93:
              v76 = v6;
              sub_A08C50((__int64)&v85, 2 * v9);
              if ( !(_DWORD)v88 )
                goto LABEL_129;
              v68 = *((_DWORD *)v15 + 2);
              v6 = v76;
              v69 = (v88 - 1) & (37 * v68);
              v21 = (_DWORD *)(v86 + 4LL * v69);
              v9 = (unsigned int)*v21;
              v22 = v87 + 1;
              if ( v68 != (_DWORD)v9 )
              {
                v70 = 1;
                v71 = 0;
                while ( (_DWORD)v9 != -1 )
                {
                  if ( (_DWORD)v9 == -2 && !v71 )
                    v71 = v21;
                  v69 = (v88 - 1) & (v70 + v69);
                  v21 = (_DWORD *)(v86 + 4LL * v69);
                  v9 = (unsigned int)*v21;
                  if ( v68 == (_DWORD)v9 )
                    goto LABEL_17;
                  ++v70;
                }
LABEL_105:
                if ( v71 )
                  v21 = v71;
              }
LABEL_17:
              LODWORD(v87) = v22;
              if ( *v21 != -1 )
                --HIDWORD(v87);
              v23 = *((unsigned int *)v15 + 2);
              *v21 = v23;
              v24 = *(_QWORD *)(*a3 + 8 * v23);
              v25 = *(unsigned int *)(a4 + 8);
              if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
              {
                v80 = v6;
                sub_C8D5F0(a4, v75, v25 + 1, 8u, (__int64)v21, v9);
                v25 = *(unsigned int *)(a4 + 8);
                v6 = v80;
              }
              *(_QWORD *)(*(_QWORD *)a4 + 8 * v25) = v24;
              ++*(_DWORD *)(a4 + 8);
              if ( (*(_DWORD *)(v11 + 4) & 0x7FFFFFF) != 0 )
              {
                v26 = 0;
                v27 = *((_DWORD *)a1 + 2);
                do
                {
                  v28 = v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF);
                  if ( (*(_BYTE *)(v11 + 7) & 0x40) != 0 )
                    v28 = *(_QWORD *)(v11 - 8);
                  if ( v6 == *(_QWORD *)(v28 + v26) )
                  {
                    if ( v27 != 1 )
                    {
                      for ( i = 1; i != v27; ++i )
                      {
                        v58 = *(_QWORD *)(**(_QWORD **)(v24 + 48) + 8LL * i);
                        if ( (*(_BYTE *)(v58 + 7) & 0x40) != 0 )
                          v57 = *(_QWORD *)(v58 - 8);
                        else
                          v57 = v58 - 32LL * (*(_DWORD *)(v58 + 4) & 0x7FFFFFF);
                        if ( *(_QWORD *)(v57 + v26) != *(_QWORD *)(*a1 + 8LL * i) )
                          goto LABEL_54;
                      }
                    }
                  }
                  else if ( v27 != 1 )
                  {
                    for ( j = 1; j != v27; ++j )
                    {
                      v31 = *(_QWORD *)(**(_QWORD **)(v24 + 48) + 8LL * j);
                      if ( (*(_BYTE *)(v31 + 7) & 0x40) != 0 )
                        v30 = *(_QWORD *)(v31 - 8);
                      else
                        v30 = v31 - 32LL * (*(_DWORD *)(v31 + 4) & 0x7FFFFFF);
                      if ( *(_QWORD *)(v30 + v26) == *(_QWORD *)(*a1 + 8LL * j) )
                        goto LABEL_54;
                    }
                  }
                  v26 += 32;
                }
                while ( 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF) != v26 );
              }
              v10 = v86;
              v9 = (unsigned int)v88;
            }
          }
        }
      }
LABEL_3:
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        goto LABEL_34;
    }
  }
  v10 = 0;
  LODWORD(v9) = 0;
LABEL_34:
  v32 = a1;
  v84 = *((_DWORD *)a1 + 2);
  if ( v84 == 1 )
  {
LABEL_58:
    v51 = 0;
    goto LABEL_55;
  }
  v33 = *v32;
  v34 = 1;
  v35 = v10;
  v82 = v9 - 1;
  while ( 1 )
  {
    v36 = *(_QWORD *)(v33 + 8LL * v34);
    v37 = *(_QWORD *)(v36 + 16);
    if ( v37 )
      break;
LABEL_56:
    if ( v84 == ++v34 )
    {
      v10 = v35;
      goto LABEL_58;
    }
  }
  while ( 1 )
  {
    v38 = *(_QWORD *)(v37 + 24);
    if ( *(_BYTE *)v38 > 0x1Cu && *(_QWORD *)(v38 + 40) == *(_QWORD *)(v36 + 40) )
    {
      v39 = *(unsigned int *)(a2 + 24);
      v40 = *(_QWORD *)(a2 + 8);
      if ( (_DWORD)v39 )
      {
        v41 = (v39 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
        v42 = (__int64 *)(v40 + 16LL * v41);
        v43 = *v42;
        if ( v38 != *v42 )
        {
          v53 = 1;
          while ( v43 != -4096 )
          {
            v41 = (v39 - 1) & (v53 + v41);
            v79 = v53 + 1;
            v42 = (__int64 *)(v40 + 16LL * v41);
            v43 = *v42;
            if ( v38 == *v42 )
              goto LABEL_43;
            v53 = v79;
          }
          goto LABEL_38;
        }
LABEL_43:
        if ( v42 != (__int64 *)(v40 + 16 * v39) )
        {
          if ( !(_DWORD)v9 )
          {
            ++v85;
            goto LABEL_77;
          }
          v44 = *((_DWORD *)v42 + 2);
          v45 = 1;
          v46 = 0;
          v47 = v82 & (37 * v44);
          v48 = (_DWORD *)(v35 + 4LL * v47);
          v49 = *v48;
          if ( *v48 != v44 )
            break;
        }
      }
    }
LABEL_38:
    v37 = *(_QWORD *)(v37 + 8);
    if ( !v37 )
      goto LABEL_56;
  }
  while ( v49 != -1 )
  {
    if ( v49 != -2 || v46 )
      v48 = v46;
    v47 = v82 & (v45 + v47);
    v49 = *(_DWORD *)(v35 + 4LL * v47);
    if ( v44 == v49 )
      goto LABEL_38;
    ++v45;
    v46 = v48;
    v48 = (_DWORD *)(v35 + 4LL * v47);
  }
  if ( !v46 )
    v46 = v48;
  ++v85;
  v50 = v87 + 1;
  if ( 4 * ((int)v87 + 1) < (unsigned int)(3 * v9) )
  {
    if ( (int)v9 - (v50 + HIDWORD(v87)) > (unsigned int)v9 >> 3 )
      goto LABEL_51;
    sub_A08C50((__int64)&v85, v9);
    if ( (_DWORD)v88 )
    {
      v64 = *((_DWORD *)v42 + 2);
      v63 = 0;
      v65 = 1;
      v66 = (v88 - 1) & (37 * v64);
      v46 = (_DWORD *)(v86 + 4LL * v66);
      v67 = *v46;
      v50 = v87 + 1;
      if ( *v46 != v64 )
      {
        while ( v67 != -1 )
        {
          if ( !v63 && v67 == -2 )
            v63 = v46;
          v66 = (v88 - 1) & (v65 + v66);
          v46 = (_DWORD *)(v86 + 4LL * v66);
          v67 = *v46;
          if ( v64 == *v46 )
            goto LABEL_51;
          ++v65;
        }
        goto LABEL_81;
      }
      goto LABEL_51;
    }
LABEL_128:
    LODWORD(v87) = v87 + 1;
    BUG();
  }
LABEL_77:
  sub_A08C50((__int64)&v85, 2 * v9);
  if ( !(_DWORD)v88 )
    goto LABEL_128;
  v59 = *((_DWORD *)v42 + 2);
  v60 = (v88 - 1) & (37 * v59);
  v46 = (_DWORD *)(v86 + 4LL * v60);
  v61 = *v46;
  v50 = v87 + 1;
  if ( *v46 != v59 )
  {
    v62 = 1;
    v63 = 0;
    while ( v61 != -1 )
    {
      if ( v61 == -2 && !v63 )
        v63 = v46;
      v60 = (v88 - 1) & (v62 + v60);
      v46 = (_DWORD *)(v86 + 4LL * v60);
      v61 = *v46;
      if ( v59 == *v46 )
        goto LABEL_51;
      ++v62;
    }
LABEL_81:
    if ( v63 )
      v46 = v63;
  }
LABEL_51:
  LODWORD(v87) = v50;
  if ( *v46 != -1 )
    --HIDWORD(v87);
  *v46 = *((_DWORD *)v42 + 2);
LABEL_54:
  LODWORD(v9) = v88;
  v10 = v86;
  v51 = 1;
LABEL_55:
  sub_C7D6A0(v10, 4LL * (unsigned int)v9, 4);
  return v51;
}
