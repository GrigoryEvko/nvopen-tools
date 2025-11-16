// Function: sub_29B1900
// Address: 0x29b1900
//
__int64 *__fastcall sub_29B1900(__int64 a1)
{
  __int64 *result; // rax
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // r9
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rsi
  int v12; // eax
  int v13; // edx
  unsigned int v14; // eax
  __int64 v15; // r8
  const char *v16; // rax
  int v17; // r13d
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rbx
  _QWORD *v21; // r15
  __int64 v22; // rax
  char v23; // dh
  __int64 v24; // rsi
  char v25; // al
  __int64 v26; // rcx
  unsigned int *v27; // r15
  unsigned int *v28; // r14
  int v29; // eax
  unsigned int v30; // ecx
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // r13
  __int64 v37; // rdx
  int v38; // eax
  unsigned int *v39; // r13
  unsigned int *v40; // r15
  unsigned int v41; // esi
  __int64 v42; // r13
  int v43; // eax
  int v44; // eax
  unsigned int v45; // edx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rdx
  unsigned __int64 v49; // rdi
  __int64 v50; // rax
  int v51; // r10d
  __int64 v52; // r15
  __int64 v53; // rdx
  __int64 v54; // r13
  __int64 v55; // rax
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rbx
  __int64 v59; // rbx
  __int64 v60; // r15
  __int64 v61; // rax
  signed __int64 v62; // r15
  char *v63; // rcx
  __int64 v64; // rdx
  int v65; // eax
  const char *v66; // r15
  const char *v67; // r12
  const char *v68; // r15
  int v69; // edx
  __int64 v70; // rax
  __int64 v71; // rdi
  int v72; // esi
  unsigned int v73; // edx
  __int64 v74; // r8
  unsigned __int64 v75; // rdx
  int v76; // eax
  __int64 v77; // rdi
  __int64 v78; // rdx
  __int64 v79; // r15
  unsigned __int16 v80; // bx
  _QWORD *v81; // rdi
  int v82; // r9d
  __int64 *v83; // [rsp+0h] [rbp-F0h]
  __int64 v84; // [rsp+10h] [rbp-E0h]
  __int64 *v85; // [rsp+18h] [rbp-D8h]
  __int64 v86; // [rsp+20h] [rbp-D0h]
  __int64 v87; // [rsp+30h] [rbp-C0h]
  __int64 v88; // [rsp+38h] [rbp-B8h]
  __int64 v89; // [rsp+40h] [rbp-B0h]
  __int64 v90; // [rsp+40h] [rbp-B0h]
  __int64 v91; // [rsp+40h] [rbp-B0h]
  __int64 v92; // [rsp+58h] [rbp-98h] BYREF
  __int64 v93; // [rsp+60h] [rbp-90h] BYREF
  unsigned __int16 v94; // [rsp+68h] [rbp-88h]
  unsigned int *v95; // [rsp+70h] [rbp-80h] BYREF
  __int64 v96; // [rsp+78h] [rbp-78h]
  _BYTE v97[16]; // [rsp+80h] [rbp-70h] BYREF
  const char *v98; // [rsp+90h] [rbp-60h] BYREF
  __int64 v99; // [rsp+98h] [rbp-58h]
  _QWORD v100[2]; // [rsp+A0h] [rbp-50h] BYREF
  __int16 v101; // [rsp+B0h] [rbp-40h]

  result = *(__int64 **)(a1 + 104);
  v85 = result;
  v83 = &result[*(unsigned int *)(a1 + 112)];
  if ( result == v83 )
    return result;
  v2 = a1;
  do
  {
    v3 = v2;
    v4 = *v85;
    v92 = 0;
    v84 = v4;
    v5 = sub_AA5930(v4);
    v88 = v6;
    if ( v5 == v6 )
      goto LABEL_48;
    do
    {
      v95 = (unsigned int *)v97;
      v96 = 0x200000000LL;
      v7 = *(unsigned int *)(v5 + 4);
      if ( (v7 & 0x7FFFFFF) == 0 )
        goto LABEL_43;
      v8 = 0;
      v9 = 0;
      do
      {
        v10 = *(_QWORD *)(v3 + 64);
        v11 = *(_QWORD *)(*(_QWORD *)(v5 - 8) + 32LL * *(unsigned int *)(v5 + 72) + 8 * v8);
        v12 = *(_DWORD *)(v3 + 80);
        if ( v12 )
        {
          v13 = v12 - 1;
          v14 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v15 = *(_QWORD *)(v10 + 8LL * v14);
          if ( v11 == v15 )
          {
LABEL_8:
            if ( v9 + 1 > (unsigned __int64)HIDWORD(v96) )
            {
              sub_C8D5F0((__int64)&v95, v97, v9 + 1, 4u, v15, v7);
              v9 = (unsigned int)v96;
            }
            v95[v9] = v8;
            v9 = (unsigned int)(v96 + 1);
            LODWORD(v96) = v96 + 1;
            v7 = *(unsigned int *)(v5 + 4);
          }
          else
          {
            v51 = 1;
            while ( v15 != -4096 )
            {
              v14 = v13 & (v51 + v14);
              v15 = *(_QWORD *)(v10 + 8LL * v14);
              if ( v11 == v15 )
                goto LABEL_8;
              ++v51;
            }
          }
        }
        ++v8;
      }
      while ( ((unsigned int)v7 & 0x7FFFFFF) > (unsigned int)v8 );
      if ( (unsigned int)v9 <= 1 )
      {
        v49 = (unsigned __int64)v95;
        if ( v95 != (unsigned int *)v97 )
          goto LABEL_42;
      }
      else
      {
        if ( !v92 )
        {
          v52 = *(_QWORD *)(v84 + 72);
          v98 = sub_BD5D20(v84);
          v101 = 773;
          v99 = v53;
          v100[0] = ".split";
          v54 = sub_AA48A0(v84);
          v55 = sub_22077B0(0x50u);
          v58 = v55;
          if ( v55 )
            sub_AA4D50(v55, v54, (__int64)&v98, v52, v84);
          v92 = v58;
          *(_BYTE *)(v58 + 40) = *(_BYTE *)(v84 + 40);
          v59 = *(_QWORD *)(v84 + 16);
          if ( v59 )
          {
            while ( (unsigned __int8)(**(_BYTE **)(v59 + 24) - 30) > 0xAu )
            {
              v59 = *(_QWORD *)(v59 + 8);
              if ( !v59 )
                goto LABEL_88;
            }
            v60 = 0;
            v98 = (const char *)v100;
            v99 = 0x400000000LL;
            v61 = v59;
            while ( 1 )
            {
              v61 = *(_QWORD *)(v61 + 8);
              if ( !v61 )
                break;
              while ( (unsigned __int8)(**(_BYTE **)(v61 + 24) - 30) <= 0xAu )
              {
                v61 = *(_QWORD *)(v61 + 8);
                ++v60;
                if ( !v61 )
                  goto LABEL_64;
              }
            }
LABEL_64:
            v62 = v60 + 1;
            v63 = (char *)v100;
            if ( v62 > 4 )
            {
              sub_C8D5F0((__int64)&v98, v100, v62, 8u, v56, v57);
              v63 = (char *)&v98[8 * (unsigned int)v99];
            }
            v64 = *(_QWORD *)(v59 + 24);
LABEL_69:
            if ( v63 )
              *(_QWORD *)v63 = *(_QWORD *)(v64 + 40);
            while ( 1 )
            {
              v59 = *(_QWORD *)(v59 + 8);
              if ( !v59 )
                break;
              v64 = *(_QWORD *)(v59 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v64 - 30) <= 0xAu )
              {
                v63 += 8;
                goto LABEL_69;
              }
            }
            v65 = v99 + v62;
            v66 = &v98[8 * (unsigned int)(v99 + v62)];
            LODWORD(v99) = v65;
            if ( v98 != v66 )
            {
              v91 = v5;
              v67 = &v98[8 * v65];
              v68 = v98;
              do
              {
                v69 = *(_DWORD *)(v3 + 80);
                v70 = *(_QWORD *)v68;
                v71 = *(_QWORD *)(v3 + 64);
                if ( v69 )
                {
                  v72 = v69 - 1;
                  v73 = (v69 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
                  v74 = *(_QWORD *)(v71 + 8LL * v73);
                  if ( v70 == v74 )
                  {
LABEL_76:
                    v75 = *(_QWORD *)(v70 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                    if ( v75 == v70 + 48 )
                    {
                      v77 = 0;
                    }
                    else
                    {
                      if ( !v75 )
                        BUG();
                      v76 = *(unsigned __int8 *)(v75 - 24);
                      v77 = 0;
                      v78 = v75 - 24;
                      if ( (unsigned int)(v76 - 30) < 0xB )
                        v77 = v78;
                    }
                    sub_BD2ED0(v77, v84, v92);
                  }
                  else
                  {
                    v82 = 1;
                    while ( v74 != -4096 )
                    {
                      v73 = v72 & (v82 + v73);
                      v74 = *(_QWORD *)(v71 + 8LL * v73);
                      if ( v70 == v74 )
                        goto LABEL_76;
                      ++v82;
                    }
                  }
                }
                v68 += 8;
              }
              while ( v67 != v68 );
              v5 = v91;
            }
          }
          else
          {
LABEL_88:
            v98 = (const char *)v100;
            v99 = 0x400000000LL;
          }
          sub_B43C20((__int64)&v93, v92);
          v79 = v93;
          v80 = v94;
          v81 = sub_BD2C40(72, 1u);
          if ( v81 )
            sub_B4C8F0((__int64)v81, v84, 1u, v79, v80);
          sub_29B0C40(v3 + 56, &v92);
          if ( v98 != (const char *)v100 )
            _libc_free((unsigned __int64)v98);
        }
        v16 = sub_BD5D20(v5);
        v17 = v96;
        v98 = v16;
        v101 = 773;
        v99 = v18;
        v100[0] = ".ce";
        v89 = *(_QWORD *)(v5 + 8);
        v19 = sub_BD2DA0(80);
        v20 = v19;
        if ( v19 )
        {
          v21 = (_QWORD *)v19;
          sub_B44260(v19, v89, 55, 0x8000000u, 0, 0);
          *(_DWORD *)(v20 + 72) = v17;
          sub_BD6B50((unsigned __int8 *)v20, &v98);
          sub_BD2A10(v20, *(_DWORD *)(v20 + 72), 1);
        }
        else
        {
          v21 = 0;
        }
        v22 = sub_AA4FF0(v92);
        v26 = v87;
        v24 = v22;
        v25 = 0;
        LOBYTE(v26) = 1;
        if ( v24 )
          v25 = v23;
        BYTE1(v26) = v25;
        v87 = v26;
        sub_B44220(v21, v24, v26);
        v27 = v95;
        if ( &v95[(unsigned int)v96] != v95 )
        {
          v86 = v3;
          v28 = &v95[(unsigned int)v96];
          do
          {
            v34 = *v27;
            v35 = *(_QWORD *)(v5 - 8);
            v36 = *(_QWORD *)(v35 + 32 * v34);
            v37 = *(_QWORD *)(v35 + 32LL * *(unsigned int *)(v5 + 72) + 8 * v34);
            v38 = *(_DWORD *)(v20 + 4) & 0x7FFFFFF;
            if ( v38 == *(_DWORD *)(v20 + 72) )
            {
              v90 = v37;
              sub_B48D90(v20);
              v37 = v90;
              v38 = *(_DWORD *)(v20 + 4) & 0x7FFFFFF;
            }
            v29 = (v38 + 1) & 0x7FFFFFF;
            v30 = v29 | *(_DWORD *)(v20 + 4) & 0xF8000000;
            v31 = *(_QWORD *)(v20 - 8) + 32LL * (unsigned int)(v29 - 1);
            *(_DWORD *)(v20 + 4) = v30;
            if ( *(_QWORD *)v31 )
            {
              v32 = *(_QWORD *)(v31 + 8);
              **(_QWORD **)(v31 + 16) = v32;
              if ( v32 )
                *(_QWORD *)(v32 + 16) = *(_QWORD *)(v31 + 16);
            }
            *(_QWORD *)v31 = v36;
            if ( v36 )
            {
              v33 = *(_QWORD *)(v36 + 16);
              *(_QWORD *)(v31 + 8) = v33;
              if ( v33 )
                *(_QWORD *)(v33 + 16) = v31 + 8;
              *(_QWORD *)(v31 + 16) = v36 + 16;
              *(_QWORD *)(v36 + 16) = v31;
            }
            ++v27;
            *(_QWORD *)(*(_QWORD *)(v20 - 8)
                      + 32LL * *(unsigned int *)(v20 + 72)
                      + 8LL * ((*(_DWORD *)(v20 + 4) & 0x7FFFFFFu) - 1)) = v37;
          }
          while ( v28 != v27 );
          v39 = v95;
          v3 = v86;
          v40 = &v95[(unsigned int)v96];
          if ( v95 != v40 )
          {
            do
            {
              v41 = *--v40;
              sub_B48BF0(v5, v41, 0);
            }
            while ( v39 != v40 );
          }
        }
        v42 = v92;
        v43 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
        if ( v43 == *(_DWORD *)(v5 + 72) )
        {
          sub_B48D90(v5);
          v43 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
        }
        v44 = (v43 + 1) & 0x7FFFFFF;
        v45 = v44 | *(_DWORD *)(v5 + 4) & 0xF8000000;
        v46 = *(_QWORD *)(v5 - 8) + 32LL * (unsigned int)(v44 - 1);
        *(_DWORD *)(v5 + 4) = v45;
        if ( *(_QWORD *)v46 )
        {
          v47 = *(_QWORD *)(v46 + 8);
          **(_QWORD **)(v46 + 16) = v47;
          if ( v47 )
            *(_QWORD *)(v47 + 16) = *(_QWORD *)(v46 + 16);
        }
        *(_QWORD *)v46 = v20;
        if ( v20 )
        {
          v48 = *(_QWORD *)(v20 + 16);
          *(_QWORD *)(v46 + 8) = v48;
          if ( v48 )
            *(_QWORD *)(v48 + 16) = v46 + 8;
          *(_QWORD *)(v46 + 16) = v20 + 16;
          *(_QWORD *)(v20 + 16) = v46;
        }
        *(_QWORD *)(*(_QWORD *)(v5 - 8)
                  + 32LL * *(unsigned int *)(v5 + 72)
                  + 8LL * ((*(_DWORD *)(v5 + 4) & 0x7FFFFFFu) - 1)) = v42;
        v49 = (unsigned __int64)v95;
        if ( v95 != (unsigned int *)v97 )
LABEL_42:
          _libc_free(v49);
      }
LABEL_43:
      v50 = *(_QWORD *)(v5 + 32);
      if ( !v50 )
        BUG();
      v5 = 0;
      if ( *(_BYTE *)(v50 - 24) == 84 )
        v5 = v50 - 24;
    }
    while ( v88 != v5 );
    v2 = v3;
LABEL_48:
    result = ++v85;
  }
  while ( v83 != v85 );
  return result;
}
