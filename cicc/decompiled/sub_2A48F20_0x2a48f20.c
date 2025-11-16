// Function: sub_2A48F20
// Address: 0x2a48f20
//
__int64 __fastcall sub_2A48F20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  int v8; // r12d
  unsigned int v9; // r14d
  int v10; // esi
  __int64 *v11; // r8
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // r9
  int *v15; // rdx
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // r15
  unsigned int v19; // esi
  char v20; // r10
  char v21; // cl
  __int64 v22; // r15
  __int64 v23; // r13
  __int64 *v24; // rdi
  int v25; // edx
  unsigned int v26; // r12d
  unsigned int v27; // r9d
  __int64 v28; // r8
  int v29; // r14d
  __int64 v30; // rsi
  __int64 v31; // rbx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 *v35; // rax
  __int64 v36; // rax
  unsigned int v37; // eax
  __int64 *v38; // rdx
  unsigned int v39; // ecx
  unsigned int v40; // r8d
  unsigned int v41; // r11d
  int v42; // r10d
  __int64 *v43; // rsi
  int v44; // edi
  unsigned int v45; // ecx
  __int64 v46; // r8
  int v47; // r9d
  __int64 *v48; // rax
  __int64 *v49; // rdi
  int v50; // esi
  unsigned int v51; // ecx
  __int64 v52; // r8
  int v53; // r9d
  unsigned int v54; // esi
  __int64 v55; // r8
  unsigned int i; // r10d
  __int64 *v57; // rdx
  __int64 v58; // rcx
  unsigned int v59; // r10d
  int v60; // esi
  __int64 v61; // r9
  int v62; // esi
  __int64 *v63; // rax
  unsigned int k; // ecx
  __int64 v65; // rdi
  int v66; // ecx
  int v67; // edi
  int v68; // ecx
  __int64 v69; // rdi
  int v70; // ecx
  unsigned int j; // r11d
  __int64 v72; // rsi
  int v73; // r11d
  unsigned __int64 v74; // [rsp+8h] [rbp-188h]
  __int64 v77; // [rsp+20h] [rbp-170h]
  int v78; // [rsp+20h] [rbp-170h]
  __int64 v80; // [rsp+28h] [rbp-168h]
  __int64 *v81; // [rsp+28h] [rbp-168h]
  unsigned __int8 *v83; // [rsp+40h] [rbp-150h]
  __int64 v84; // [rsp+48h] [rbp-148h]
  __int64 v85; // [rsp+50h] [rbp-140h] BYREF
  __int64 v86; // [rsp+58h] [rbp-138h]
  __int64 *v87; // [rsp+60h] [rbp-130h] BYREF
  unsigned int v88; // [rsp+68h] [rbp-128h]
  char v89; // [rsp+160h] [rbp-30h] BYREF

  v83 = **(unsigned __int8 ***)(a2 - 8);
  result = *v83;
  if ( (unsigned __int8)result <= 0x1Cu && (_BYTE)result != 22 )
    return result;
  result = *((_QWORD *)v83 + 2);
  if ( result )
  {
    if ( !*(_QWORD *)(result + 8) )
      return result;
  }
  v5 = (__int64 *)&v87;
  v85 = 0;
  v86 = 1;
  do
  {
    *v5 = -4096;
    v5 += 2;
  }
  while ( v5 != (__int64 *)&v89 );
  v6 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 != a3 + 48 )
  {
    if ( !v6 )
      BUG();
    v7 = v6 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 <= 0xA )
    {
      v8 = sub_B46E30(v7);
      if ( v8 )
      {
        v9 = 0;
        while ( 1 )
        {
          v17 = sub_B46EC0(v7, v9);
          v18 = v17;
          if ( (v86 & 1) != 0 )
          {
            v10 = 15;
            v11 = (__int64 *)&v87;
          }
          else
          {
            v19 = v88;
            v11 = v87;
            if ( !v88 )
            {
              v37 = v86;
              ++v85;
              v38 = 0;
              v39 = ((unsigned int)v86 >> 1) + 1;
              goto LABEL_37;
            }
            v10 = v88 - 1;
          }
          v12 = v10 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v13 = &v11[2 * v12];
          v14 = *v13;
          if ( v18 != *v13 )
            break;
LABEL_13:
          v15 = (int *)(v13 + 1);
          v16 = *((_DWORD *)v13 + 2) + 1;
LABEL_14:
          ++v9;
          *v15 = v16;
          if ( v8 == v9 )
            goto LABEL_21;
        }
        v42 = 1;
        v38 = 0;
        while ( v14 != -4096 )
        {
          if ( v14 == -8192 && !v38 )
            v38 = v13;
          v12 = v10 & (v42 + v12);
          v13 = &v11[2 * v12];
          v14 = *v13;
          if ( v18 == *v13 )
            goto LABEL_13;
          ++v42;
        }
        v40 = 48;
        v19 = 16;
        if ( !v38 )
          v38 = v13;
        v37 = v86;
        ++v85;
        v39 = ((unsigned int)v86 >> 1) + 1;
        if ( (v86 & 1) == 0 )
        {
          v19 = v88;
LABEL_37:
          v40 = 3 * v19;
        }
        if ( v40 <= 4 * v39 )
        {
          sub_2796C10((__int64)&v85, 2 * v19);
          if ( (v86 & 1) != 0 )
          {
            v44 = 15;
            v43 = (__int64 *)&v87;
          }
          else
          {
            v43 = v87;
            if ( !v88 )
              goto LABEL_141;
            v44 = v88 - 1;
          }
          v37 = v86;
          v45 = v44 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v38 = &v43[2 * v45];
          v46 = *v38;
          if ( v18 == *v38 )
            goto LABEL_40;
          v47 = 1;
          v48 = 0;
          while ( v46 != -4096 )
          {
            if ( !v48 && v46 == -8192 )
              v48 = v38;
            v45 = v44 & (v47 + v45);
            v38 = &v43[2 * v45];
            v46 = *v38;
            if ( v18 == *v38 )
              goto LABEL_62;
            ++v47;
          }
        }
        else
        {
          if ( v19 - HIDWORD(v86) - v39 > v19 >> 3 )
            goto LABEL_40;
          sub_2796C10((__int64)&v85, v19);
          if ( (v86 & 1) != 0 )
          {
            v50 = 15;
            v49 = (__int64 *)&v87;
          }
          else
          {
            v49 = v87;
            if ( !v88 )
            {
LABEL_141:
              LODWORD(v86) = (2 * ((unsigned int)v86 >> 1) + 2) | v86 & 1;
              BUG();
            }
            v50 = v88 - 1;
          }
          v37 = v86;
          v51 = v50 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v38 = &v49[2 * v51];
          v52 = *v38;
          if ( v18 == *v38 )
            goto LABEL_40;
          v53 = 1;
          v48 = 0;
          while ( v52 != -4096 )
          {
            if ( v52 == -8192 && !v48 )
              v48 = v38;
            v51 = v50 & (v53 + v51);
            v38 = &v49[2 * v51];
            v52 = *v38;
            if ( v18 == *v38 )
              goto LABEL_62;
            ++v53;
          }
        }
        if ( v48 )
          v38 = v48;
LABEL_62:
        v37 = v86;
LABEL_40:
        LODWORD(v86) = (2 * (v37 >> 1) + 2) | v37 & 1;
        if ( *v38 != -4096 )
          --HIDWORD(v86);
        *v38 = v18;
        v16 = 1;
        v15 = (int *)(v38 + 1);
        *v15 = 0;
        goto LABEL_14;
      }
    }
  }
LABEL_21:
  v21 = v86;
  result = (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1;
  if ( (_DWORD)result != 1 )
  {
    v84 = (unsigned int)(result - 1);
    v22 = 0;
    v23 = a2;
    v74 = (unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32;
    while ( 1 )
    {
      result = 32;
      if ( (_DWORD)v22 != -2 )
        result = 32LL * (unsigned int)(2 * v22 + 3);
      v30 = *(_QWORD *)(v23 - 8);
      ++v22;
      v31 = *(_QWORD *)(v30 + result);
      v20 = v21 & 1;
      if ( (v21 & 1) != 0 )
        break;
      v24 = v87;
      if ( v88 )
      {
        v25 = v88 - 1;
LABEL_24:
        v26 = ((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4);
        v27 = v25 & v26;
        result = (__int64)&v24[2 * (v25 & v26)];
        v28 = *(_QWORD *)result;
        if ( v31 == *(_QWORD *)result )
        {
LABEL_25:
          v29 = *(_DWORD *)(result + 8);
          if ( v29 == 1 )
          {
            v77 = *(_QWORD *)(v23 + 40);
            v80 = *(_QWORD *)(v30 + 32LL * (unsigned int)(2 * v22));
            v32 = sub_22077B0(0x58u);
            if ( v32 )
            {
              v35 = *(__int64 **)(v23 - 8);
              *(_QWORD *)(v32 + 80) = v23;
              v36 = *v35;
              *(_QWORD *)(v32 + 32) = v83;
              *(_QWORD *)(v32 + 8) = 0;
              *(_QWORD *)(v32 + 48) = v36;
              *(_QWORD *)(v32 + 16) = 0;
              *(_DWORD *)(v32 + 24) = 2;
              *(_QWORD *)(v32 + 56) = v77;
              *(_QWORD *)(v32 + 64) = v31;
              *(_QWORD *)v32 = &unk_4A22E20;
              *(_QWORD *)(v32 + 72) = v80;
            }
            sub_2A481C0(a1, a4, (__int64)v83, v32, v33, v34);
            result = sub_AA54C0(v31);
            if ( !result )
            {
              v54 = *(_DWORD *)(a1 + 1640);
              v55 = a1 + 1616;
              if ( v54 )
              {
                v78 = 1;
                v81 = 0;
                for ( i = (((0xBF58476D1CE4E5B9LL * (v74 | v26)) >> 31) ^ (484763065 * (v74 | v26))) & (v54 - 1);
                      ;
                      i = (v54 - 1) & v59 )
                {
                  v57 = (__int64 *)(*(_QWORD *)(a1 + 1624) + 16LL * i);
                  v58 = *v57;
                  if ( a3 == *v57 && v31 == v57[1] )
                    goto LABEL_35;
                  if ( v58 == -4096 )
                  {
                    if ( v57[1] == -4096 )
                    {
                      if ( v81 )
                        v57 = v81;
                      ++*(_QWORD *)(a1 + 1616);
                      v67 = *(_DWORD *)(a1 + 1632) + 1;
                      if ( 4 * v67 < 3 * v54 )
                      {
                        if ( v54 - *(_DWORD *)(a1 + 1636) - v67 > v54 >> 3 )
                        {
LABEL_104:
                          *(_DWORD *)(a1 + 1632) = v67;
                          if ( *v57 != -4096 || v57[1] != -4096 )
                            --*(_DWORD *)(a1 + 1636);
                          result = a3;
                          v57[1] = v31;
                          *v57 = a3;
                          goto LABEL_35;
                        }
                        sub_2884B10(v55, v54);
                        v68 = *(_DWORD *)(a1 + 1640);
                        if ( v68 )
                        {
                          v69 = *(_QWORD *)(a1 + 1624);
                          v70 = v68 - 1;
                          v63 = 0;
                          for ( j = v70 & (((0xBF58476D1CE4E5B9LL * (v74 | v26)) >> 31) ^ (484763065 * (v74 | v26)));
                                ;
                                j = v70 & v73 )
                          {
                            v57 = (__int64 *)(v69 + 16LL * j);
                            v72 = *v57;
                            if ( a3 == *v57 && v31 == v57[1] )
                              break;
                            if ( v72 == -4096 )
                            {
                              if ( v57[1] == -4096 )
                                goto LABEL_135;
                            }
                            else if ( v72 == -8192 && v57[1] == -8192 && !v63 )
                            {
                              v63 = (__int64 *)(v69 + 16LL * j);
                            }
                            v73 = v29 + j;
                            ++v29;
                          }
LABEL_131:
                          v67 = *(_DWORD *)(a1 + 1632) + 1;
                          goto LABEL_104;
                        }
                        goto LABEL_140;
                      }
LABEL_90:
                      sub_2884B10(v55, 2 * v54);
                      v60 = *(_DWORD *)(a1 + 1640);
                      if ( v60 )
                      {
                        v61 = *(_QWORD *)(a1 + 1624);
                        v62 = v60 - 1;
                        v63 = 0;
                        for ( k = v62 & (((0xBF58476D1CE4E5B9LL * (v74 | v26)) >> 31) ^ (484763065 * (v74 | v26)));
                              ;
                              k = v62 & v66 )
                        {
                          v57 = (__int64 *)(v61 + 16LL * k);
                          v65 = *v57;
                          if ( a3 == *v57 && v31 == v57[1] )
                            break;
                          if ( v65 == -4096 )
                          {
                            if ( v57[1] == -4096 )
                            {
LABEL_135:
                              if ( v63 )
                                v57 = v63;
                              v67 = *(_DWORD *)(a1 + 1632) + 1;
                              goto LABEL_104;
                            }
                          }
                          else if ( v65 == -8192 && v57[1] == -8192 && !v63 )
                          {
                            v63 = (__int64 *)(v61 + 16LL * k);
                          }
                          v66 = v29 + k;
                          ++v29;
                        }
                        goto LABEL_131;
                      }
LABEL_140:
                      ++*(_DWORD *)(a1 + 1632);
                      BUG();
                    }
                  }
                  else if ( v58 == -8192 && v57[1] == -8192 )
                  {
                    if ( v81 )
                      v57 = v81;
                    v81 = v57;
                  }
                  v59 = v78 + i;
                  ++v78;
                }
              }
              ++*(_QWORD *)(a1 + 1616);
              goto LABEL_90;
            }
LABEL_35:
            v21 = v86;
            v20 = v86 & 1;
          }
        }
        else
        {
          result = 1;
          while ( v28 != -4096 )
          {
            v41 = result + 1;
            v27 = v25 & (result + v27);
            result = (__int64)&v24[2 * v27];
            v28 = *(_QWORD *)result;
            if ( *(_QWORD *)result == v31 )
              goto LABEL_25;
            result = v41;
          }
        }
      }
      if ( v84 == v22 )
        goto LABEL_19;
    }
    v24 = (__int64 *)&v87;
    v25 = 15;
    goto LABEL_24;
  }
  v20 = v86 & 1;
LABEL_19:
  if ( !v20 )
    return sub_C7D6A0((__int64)v87, 16LL * v88, 8);
  return result;
}
