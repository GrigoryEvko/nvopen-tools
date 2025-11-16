// Function: sub_2A6D850
// Address: 0x2a6d850
//
void __fastcall sub_2A6D850(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4)
{
  unsigned __int64 v5; // r13
  __int64 v6; // r14
  __int64 v7; // rdx
  unsigned __int64 v8; // r14
  __int64 v9; // rdx
  unsigned int v10; // r15d
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // r14
  unsigned int v13; // r13d
  unsigned int v14; // esi
  __int64 v15; // rax
  unsigned int v16; // r8d
  __int64 v17; // rdi
  int v18; // r11d
  unsigned __int64 v19; // rbx
  unsigned int i; // edx
  __int64 *v21; // r9
  __int64 v22; // r10
  unsigned __int8 *v23; // r9
  unsigned __int8 v24; // al
  unsigned __int8 *v25; // rbx
  int v26; // r10d
  unsigned __int64 v27; // r9
  unsigned int v28; // eax
  unsigned __int64 v29; // rdx
  __int64 v30; // r11
  unsigned int v31; // eax
  unsigned int v32; // eax
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  __int64 *v35; // rax
  unsigned __int8 *v36; // r15
  __int64 *v37; // rbx
  __int64 v38; // r14
  __int64 v39; // rax
  unsigned int v40; // r12d
  unsigned int v41; // esi
  __int64 v42; // r8
  int v43; // r10d
  unsigned __int64 v44; // rcx
  unsigned int j; // eax
  unsigned __int64 v46; // rdx
  __int64 v47; // r9
  unsigned __int64 v48; // r15
  unsigned __int8 *v49; // rax
  int v50; // ecx
  unsigned __int64 v51; // rdx
  int v52; // edi
  unsigned int v53; // eax
  unsigned __int64 v54; // rdx
  int v55; // eax
  __int64 *v56; // rax
  unsigned int v57; // edx
  unsigned int v58; // eax
  unsigned __int64 v59; // rdi
  int v60; // edx
  unsigned int v61; // eax
  int v62; // eax
  int v63; // eax
  unsigned int v64; // eax
  int v65; // eax
  unsigned __int64 v66; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v67; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v68; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v69; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v70; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v71; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v72; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v73; // [rsp+10h] [rbp-A0h]
  __int64 v74; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v75; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v76; // [rsp+10h] [rbp-A0h]
  __int64 v77; // [rsp+18h] [rbp-98h]
  __int64 v79; // [rsp+28h] [rbp-88h]
  unsigned __int64 v80; // [rsp+28h] [rbp-88h]
  unsigned __int8 **v81; // [rsp+30h] [rbp-80h]
  unsigned __int64 v82; // [rsp+38h] [rbp-78h]
  unsigned __int64 v83; // [rsp+40h] [rbp-70h]
  unsigned __int64 v84; // [rsp+40h] [rbp-70h]
  int v85; // [rsp+48h] [rbp-68h]
  int v86; // [rsp+48h] [rbp-68h]
  unsigned __int64 v87; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v88; // [rsp+58h] [rbp-58h]
  unsigned __int64 v89; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v90; // [rsp+68h] [rbp-48h]
  unsigned __int64 v91; // [rsp+70h] [rbp-40h]
  unsigned int v92; // [rsp+78h] [rbp-38h]

  v81 = *(unsigned __int8 ***)a3;
  if ( (*(_BYTE *)(a2 + 2) & 1) == 0 )
  {
    v5 = *(_QWORD *)(a2 + 96);
    v6 = *((_QWORD *)*v81 + 3);
    if ( (*(_BYTE *)(v6 + 2) & 1) == 0 )
    {
      v7 = *(_QWORD *)(a2 + 96);
      v8 = *(_QWORD *)(v6 + 96);
      goto LABEL_4;
    }
    goto LABEL_20;
  }
  sub_B2C6D0(a2, a2, (__int64)a3, a4);
  v5 = *(_QWORD *)(a2 + 96);
  v6 = *(_QWORD *)(**(_QWORD **)a3 + 24LL);
  if ( (*(_BYTE *)(v6 + 2) & 1) != 0 )
LABEL_20:
    sub_B2C6D0(v6, a2, (__int64)a3, a4);
  v8 = *(_QWORD *)(v6 + 96);
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    sub_B2C6D0(a2, a2, (__int64)a3, a4);
  v7 = *(_QWORD *)(a2 + 96);
LABEL_4:
  v77 = v7 + 40LL * *(_QWORD *)(a2 + 104);
  if ( v77 == v5 )
    return;
  do
  {
    v9 = *(_QWORD *)(v5 + 8);
    if ( v81 == (unsigned __int8 **)(*(_QWORD *)a3 + 16LL * a3[2]) || *v81 != (unsigned __int8 *)v8 )
    {
      if ( *(_BYTE *)(v9 + 8) != 15 )
      {
        v89 = v5;
        v35 = sub_2A686D0(a1 + 136, (__int64 *)&v89);
        v89 = v8;
        v36 = (unsigned __int8 *)v35;
        v37 = sub_2A686D0(a1 + 136, (__int64 *)&v89);
        sub_22C0090(v36);
        sub_22C05A0((__int64)v36, (unsigned __int8 *)v37);
        goto LABEL_32;
      }
      v85 = *(_DWORD *)(v9 + 12);
      if ( !v85 )
        goto LABEL_32;
      v10 = 0;
      v83 = (unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32;
      v82 = (unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32;
      v79 = a1 + 168;
      v11 = v8;
      v12 = v5;
      v13 = 0;
LABEL_10:
      v14 = *(_DWORD *)(a1 + 192);
      v89 = v12;
      v90 = v10;
      if ( !v14 )
      {
        ++*(_QWORD *)(a1 + 168);
        v87 = 0;
LABEL_93:
        v75 = v11;
        v14 *= 2;
        goto LABEL_94;
      }
      v15 = v13;
      v16 = v14 - 1;
      v17 = *(_QWORD *)(a1 + 176);
      v18 = 1;
      v19 = 0;
      for ( i = (v14 - 1) & (((0xBF58476D1CE4E5B9LL * (v13 | v83)) >> 31) ^ (484763065 * (v13 | v83))); ; i = v16 & v57 )
      {
        v21 = (__int64 *)(v17 + 56LL * i);
        v22 = *v21;
        if ( *v21 == v12 && *((_DWORD *)v21 + 2) == v10 )
        {
          v89 = v11;
          v25 = (unsigned __int8 *)(v21 + 2);
          v90 = v10;
          goto LABEL_36;
        }
        if ( v22 == -4096 )
        {
          if ( *((_DWORD *)v21 + 2) == -1 )
          {
            v63 = *(_DWORD *)(a1 + 184);
            if ( !v19 )
              v19 = v17 + 56LL * i;
            ++*(_QWORD *)(a1 + 168);
            v52 = v63 + 1;
            v87 = v19;
            if ( 4 * (v63 + 1) >= 3 * v14 )
              goto LABEL_93;
            v51 = v12;
            if ( v14 - *(_DWORD *)(a1 + 188) - v52 <= v14 >> 3 )
            {
              v75 = v11;
LABEL_94:
              sub_2A69E40(v79, v14);
              sub_2A65F30(v79, (__int64 *)&v89, &v87);
              v51 = v89;
              v19 = v87;
              v11 = v75;
              v52 = *(_DWORD *)(a1 + 184) + 1;
            }
            *(_DWORD *)(a1 + 184) = v52;
            if ( *(_QWORD *)v19 != -4096 || *(_DWORD *)(v19 + 8) != -1 )
              --*(_DWORD *)(a1 + 188);
            *(_QWORD *)v19 = v51;
            v53 = v90;
            v25 = (unsigned __int8 *)(v19 + 16);
            *(_WORD *)v25 = 0;
            *((_DWORD *)v25 - 2) = v53;
            v14 = *(_DWORD *)(a1 + 192);
            v89 = v11;
            v17 = *(_QWORD *)(a1 + 176);
            v90 = v10;
            if ( !v14 )
            {
              ++*(_QWORD *)(a1 + 168);
              v87 = 0;
LABEL_102:
              v76 = v11;
              v14 *= 2;
              goto LABEL_103;
            }
            v15 = v13;
            v16 = v14 - 1;
LABEL_36:
            v26 = 1;
            v27 = 0;
            v28 = v16 & (((0xBF58476D1CE4E5B9LL * (v82 | v15)) >> 31) ^ (484763065 * (v82 | v15)));
            while ( 2 )
            {
              v29 = v17 + 56LL * v28;
              v30 = *(_QWORD *)v29;
              if ( *(_QWORD *)v29 == v11 && *(_DWORD *)(v29 + 8) == v10 )
              {
                v23 = (unsigned __int8 *)(v29 + 16);
                goto LABEL_26;
              }
              if ( v30 != -4096 )
              {
                if ( v30 == -8192 && *(_DWORD *)(v29 + 8) == -2 && !v27 )
                  v27 = v17 + 56LL * v28;
                goto LABEL_114;
              }
              if ( *(_DWORD *)(v29 + 8) != -1 )
              {
LABEL_114:
                v58 = v26 + v28;
                ++v26;
                v28 = v16 & v58;
                continue;
              }
              break;
            }
            v62 = *(_DWORD *)(a1 + 184);
            if ( !v27 )
              v27 = v29;
            ++*(_QWORD *)(a1 + 168);
            v55 = v62 + 1;
            v87 = v27;
            if ( 4 * v55 >= 3 * v14 )
              goto LABEL_102;
            v54 = v11;
            if ( v14 - (v55 + *(_DWORD *)(a1 + 188)) <= v14 >> 3 )
            {
              v76 = v11;
LABEL_103:
              sub_2A69E40(v79, v14);
              sub_2A65F30(v79, (__int64 *)&v89, &v87);
              v54 = v89;
              v27 = v87;
              v11 = v76;
              v55 = *(_DWORD *)(a1 + 184) + 1;
            }
            *(_DWORD *)(a1 + 184) = v55;
            if ( *(_QWORD *)v27 != -4096 || *(_DWORD *)(v27 + 8) != -1 )
              --*(_DWORD *)(a1 + 188);
            *(_QWORD *)v27 = v54;
            v23 = (unsigned __int8 *)(v27 + 16);
            *((_DWORD *)v23 - 2) = v90;
            *(_WORD *)v23 = 0;
LABEL_26:
            if ( (unsigned int)*v25 - 4 <= 1 )
            {
              if ( *((_DWORD *)v25 + 8) > 0x40u )
              {
                v33 = *((_QWORD *)v25 + 3);
                if ( v33 )
                {
                  v66 = v11;
                  v70 = v23;
                  j_j___libc_free_0_0(v33);
                  v11 = v66;
                  v23 = v70;
                }
              }
              if ( *((_DWORD *)v25 + 4) > 0x40u )
              {
                v34 = *((_QWORD *)v25 + 1);
                if ( v34 )
                {
                  v67 = v11;
                  v71 = v23;
                  j_j___libc_free_0_0(v34);
                  v11 = v67;
                  v23 = v71;
                }
              }
            }
            v24 = *v23;
            *(_WORD *)v25 = *v23;
            if ( v24 > 3u )
            {
              if ( (unsigned __int8)(v24 - 4) <= 1u )
              {
                v31 = *((_DWORD *)v23 + 4);
                *((_DWORD *)v25 + 4) = v31;
                if ( v31 > 0x40 )
                {
                  v69 = v11;
                  v73 = v23;
                  sub_C43780((__int64)(v25 + 8), (const void **)v23 + 1);
                  v11 = v69;
                  v23 = v73;
                }
                else
                {
                  *((_QWORD *)v25 + 1) = *((_QWORD *)v23 + 1);
                }
                v32 = *((_DWORD *)v23 + 8);
                *((_DWORD *)v25 + 8) = v32;
                if ( v32 > 0x40 )
                {
                  v68 = v11;
                  v72 = v23;
                  sub_C43780((__int64)(v25 + 24), (const void **)v23 + 3);
                  v11 = v68;
                  v23 = v72;
                }
                else
                {
                  *((_QWORD *)v25 + 3) = *((_QWORD *)v23 + 3);
                }
                v25[1] = v23[1];
              }
            }
            else if ( v24 > 1u )
            {
              *((_QWORD *)v25 + 1) = *((_QWORD *)v23 + 1);
            }
            ++v10;
            v13 += 37;
            if ( v10 == v85 )
            {
              v5 = v12;
              v8 = v11;
              goto LABEL_32;
            }
            goto LABEL_10;
          }
        }
        else if ( v22 == -8192 && *((_DWORD *)v21 + 2) == -2 && !v19 )
        {
          v19 = v17 + 56LL * i;
        }
        v57 = v18 + i;
        ++v18;
      }
    }
    if ( *(_BYTE *)(v9 + 8) != 15 )
    {
      v89 = v5;
      v56 = sub_2A686D0(a1 + 136, (__int64 *)&v89);
      sub_2A624B0((__int64)v56, v81[1], 0);
      v81 += 2;
      goto LABEL_32;
    }
    v86 = *(_DWORD *)(v9 + 12);
    if ( !v86 )
      goto LABEL_78;
    v80 = v8;
    v38 = a1;
    v39 = a1 + 168;
    v40 = 0;
    v74 = v39;
    do
    {
      v41 = *(_DWORD *)(v38 + 192);
      v89 = v5;
      v90 = v40;
      if ( !v41 )
      {
        ++*(_QWORD *)(v38 + 168);
        v87 = 0;
LABEL_117:
        v41 *= 2;
        goto LABEL_118;
      }
      v42 = *(_QWORD *)(v38 + 176);
      v43 = 1;
      v44 = 0;
      v84 = (unsigned __int64)(((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4)) << 32;
      for ( j = (v41 - 1) & (((0xBF58476D1CE4E5B9LL * (v84 | (37 * v40))) >> 31) ^ (484763065 * (v84 | (37 * v40))));
            ;
            j = (v41 - 1) & v64 )
      {
        v46 = v42 + 56LL * j;
        v47 = *(_QWORD *)v46;
        if ( *(_QWORD *)v46 == v5 && *(_DWORD *)(v46 + 8) == v40 )
        {
          v48 = v46 + 16;
          goto LABEL_73;
        }
        if ( v47 == -4096 )
          break;
        if ( v47 == -8192 && *(_DWORD *)(v46 + 8) == -2 && !v44 )
          v44 = v42 + 56LL * j;
LABEL_135:
        v64 = v43 + j;
        ++v43;
      }
      if ( *(_DWORD *)(v46 + 8) != -1 )
        goto LABEL_135;
      v65 = *(_DWORD *)(v38 + 184);
      if ( !v44 )
        v44 = v46;
      ++*(_QWORD *)(v38 + 168);
      v60 = v65 + 1;
      v87 = v44;
      if ( 4 * (v65 + 1) >= 3 * v41 )
        goto LABEL_117;
      v59 = v5;
      if ( v41 - *(_DWORD *)(v38 + 188) - v60 <= v41 >> 3 )
      {
LABEL_118:
        sub_2A69E40(v74, v41);
        sub_2A65F30(v74, (__int64 *)&v89, &v87);
        v59 = v89;
        v44 = v87;
        v60 = *(_DWORD *)(v38 + 184) + 1;
      }
      *(_DWORD *)(v38 + 184) = v60;
      if ( *(_QWORD *)v44 != -4096 || *(_DWORD *)(v44 + 8) != -1 )
        --*(_DWORD *)(v38 + 188);
      *(_QWORD *)v44 = v59;
      v61 = v90;
      v48 = v44 + 16;
      *(_WORD *)(v44 + 16) = 0;
      *(_DWORD *)(v44 + 8) = v61;
LABEL_73:
      v49 = (unsigned __int8 *)sub_AD69F0(v81[1], v40);
      v50 = *v49;
      if ( (unsigned int)(v50 - 12) > 1 )
      {
        if ( *(_BYTE *)v48 != 2 )
        {
          if ( (_BYTE)v50 == 17 )
          {
            v88 = *((_DWORD *)v49 + 8);
            if ( v88 > 0x40 )
              sub_C43780((__int64)&v87, (const void **)v49 + 3);
            else
              v87 = *((_QWORD *)v49 + 3);
            sub_AADBC0((__int64)&v89, (__int64 *)&v87);
            sub_2A62120((char *)v48, (__int64)&v89, 0, 0, 1u);
            if ( v92 > 0x40 && v91 )
              j_j___libc_free_0_0(v91);
            if ( v90 > 0x40 && v89 )
              j_j___libc_free_0_0(v89);
            if ( v88 > 0x40 && v87 )
              j_j___libc_free_0_0(v87);
          }
          else
          {
            *(_BYTE *)v48 = 2;
            *(_QWORD *)(v48 + 8) = v49;
          }
        }
      }
      else if ( *(_BYTE *)v48 != 1 )
      {
        *(_BYTE *)v48 = 1;
      }
      ++v40;
    }
    while ( v40 != v86 );
    a1 = v38;
    v8 = v80;
LABEL_78:
    v81 += 2;
LABEL_32:
    v5 += 40LL;
    v8 += 40LL;
  }
  while ( v5 != v77 );
}
