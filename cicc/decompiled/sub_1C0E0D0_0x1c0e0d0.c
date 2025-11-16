// Function: sub_1C0E0D0
// Address: 0x1c0e0d0
//
__int64 __fastcall sub_1C0E0D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r14
  __int64 v8; // r12
  __int64 v9; // r15
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 *v17; // rbx
  __int64 v18; // rax
  unsigned int v19; // esi
  int v20; // edi
  __int64 v22; // r14
  __int64 v23; // r15
  __int64 v24; // r13
  unsigned __int64 v25; // rcx
  __int64 v26; // r9
  unsigned int v27; // edx
  unsigned __int64 *v28; // rax
  unsigned __int64 v29; // r8
  __int64 v30; // rax
  _QWORD *v31; // rdx
  _QWORD *v32; // rax
  _QWORD *v33; // r12
  __int64 v34; // rax
  __int64 v35; // rbx
  unsigned int v36; // esi
  unsigned __int64 *v37; // rdi
  int v38; // edx
  _QWORD *v39; // rax
  _QWORD *v40; // rbx
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // r12
  __int64 v46; // rax
  int *v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r14
  __int64 v51; // rax
  int v52; // r8d
  int v53; // r9d
  __int64 v54; // rcx
  __int64 v55; // rbx
  _QWORD *v56; // rax
  __int64 v57; // rdx
  unsigned int v58; // esi
  __int64 v59; // rdx
  __int64 v60; // r8
  unsigned int v61; // eax
  __int64 v62; // rbx
  __int64 v63; // rdi
  unsigned int v64; // esi
  __int64 v65; // r8
  __int64 v66; // r14
  __int64 v67; // rdx
  unsigned int v68; // eax
  __int64 *v69; // rdi
  __int64 v70; // rcx
  __int64 v71; // rdx
  __int64 v72; // rcx
  int v73; // r11d
  __int64 *v74; // r10
  int v75; // edi
  __int64 v76; // rcx
  int v77; // edi
  int v78; // r11d
  int v79; // eax
  __int64 *v80; // r10
  int v81; // r11d
  int v82; // eax
  int v83; // eax
  int v84; // r10d
  int v85; // eax
  __int64 v86; // rax
  int v87; // esi
  _QWORD *v88; // rdx
  unsigned __int64 *v89; // r12
  __int64 v90; // [rsp+10h] [rbp-E0h]
  __int64 *v91; // [rsp+10h] [rbp-E0h]
  __int64 v93; // [rsp+20h] [rbp-D0h]
  __int64 v94; // [rsp+30h] [rbp-C0h]
  __int64 v98; // [rsp+58h] [rbp-98h]
  unsigned __int64 v100; // [rsp+70h] [rbp-80h] BYREF
  __int64 v101; // [rsp+78h] [rbp-78h] BYREF
  __int64 v102; // [rsp+80h] [rbp-70h] BYREF
  __int64 *v103; // [rsp+88h] [rbp-68h] BYREF
  __int64 *v104; // [rsp+90h] [rbp-60h] BYREF
  int v105; // [rsp+98h] [rbp-58h] BYREF
  __int64 v106; // [rsp+A0h] [rbp-50h]
  int *v107; // [rsp+A8h] [rbp-48h]
  int *v108; // [rsp+B0h] [rbp-40h]
  __int64 v109; // [rsp+B8h] [rbp-38h]

  v7 = a2 + 72;
  v8 = a1;
  v9 = *(_QWORD *)(a2 + 80);
  v94 = a1 + 40;
  if ( v9 == a2 + 72 )
    return sub_1C0DCE0(v8, a2, a7);
  do
  {
    v15 = v9 - 24;
    if ( !v9 )
      v15 = 0;
    v103 = (__int64 *)v15;
    v16 = (__int64 *)sub_22077B0(80);
    v17 = v16;
    if ( v16 )
    {
      v16[4] = 0x100000000LL;
      v16[3] = (__int64)(v16 + 5);
      v16[6] = (__int64)(v16 + 8);
      v18 = (__int64)v103;
      v17[7] = 0x100000000LL;
      *v17 = v18;
      *((_BYTE *)v17 + 8) = 0;
      *(__int64 *)((char *)v17 + 12) = 0;
      v17[9] = 0;
    }
    v19 = *(_DWORD *)(v8 + 64);
    if ( v19 )
    {
      v10 = (__int64)v103;
      v11 = *(_QWORD *)(v8 + 48);
      v12 = (v19 - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
      v13 = (__int64 *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( v103 == (__int64 *)*v13 )
        goto LABEL_4;
      v73 = 1;
      v74 = 0;
      while ( v14 != -8 )
      {
        if ( !v74 && v14 == -16 )
          v74 = v13;
        v12 = (v19 - 1) & (v73 + v12);
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( v103 == (__int64 *)*v13 )
          goto LABEL_4;
        ++v73;
      }
      v75 = *(_DWORD *)(v8 + 56);
      if ( v74 )
        v13 = v74;
      ++*(_QWORD *)(v8 + 40);
      v20 = v75 + 1;
      if ( 4 * v20 < 3 * v19 )
      {
        if ( v19 - *(_DWORD *)(v8 + 60) - v20 > v19 >> 3 )
          goto LABEL_77;
        goto LABEL_12;
      }
    }
    else
    {
      ++*(_QWORD *)(v8 + 40);
    }
    v19 *= 2;
LABEL_12:
    sub_1C04E30(v94, v19);
    sub_1C09800(v94, (__int64 *)&v103, &v104);
    v13 = v104;
    v10 = (__int64)v103;
    v20 = *(_DWORD *)(v8 + 56) + 1;
LABEL_77:
    *(_DWORD *)(v8 + 56) = v20;
    if ( *v13 != -8 )
      --*(_DWORD *)(v8 + 60);
    *v13 = v10;
    v13[1] = 0;
LABEL_4:
    v13[1] = (__int64)v17;
    v9 = *(_QWORD *)(v9 + 8);
  }
  while ( v7 != v9 );
  v22 = *(_QWORD *)(a2 + 80);
  if ( v9 == v22 )
    return sub_1C0DCE0(v8, a2, a7);
  v98 = v9;
  v23 = v8;
  v24 = a6;
  while ( 2 )
  {
    v35 = v22 - 24;
    if ( !v22 )
      v35 = 0;
    v100 = v35;
    if ( (unsigned __int8)sub_1C089B0(v35, a5) )
    {
      v36 = *(_DWORD *)(v23 + 64);
      if ( !v36 )
      {
        ++*(_QWORD *)(v23 + 40);
        goto LABEL_31;
      }
      v25 = v100;
      v26 = *(_QWORD *)(v23 + 48);
      v27 = (v36 - 1) & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
      v28 = (unsigned __int64 *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( *v28 == v100 )
      {
        v30 = v28[1];
      }
      else
      {
        v78 = 1;
        v37 = 0;
        while ( v29 != -8 )
        {
          if ( v37 || v29 != -16 )
            v28 = v37;
          v27 = (v36 - 1) & (v78 + v27);
          v89 = (unsigned __int64 *)(v26 + 16LL * v27);
          v29 = *v89;
          if ( v100 == *v89 )
          {
            v30 = v89[1];
            goto LABEL_19;
          }
          ++v78;
          v37 = v28;
          v28 = (unsigned __int64 *)(v26 + 16LL * v27);
        }
        if ( !v37 )
          v37 = v28;
        v79 = *(_DWORD *)(v23 + 56);
        ++*(_QWORD *)(v23 + 40);
        v38 = v79 + 1;
        if ( 4 * (v79 + 1) >= 3 * v36 )
        {
LABEL_31:
          v36 *= 2;
LABEL_32:
          sub_1C04E30(v94, v36);
          sub_1C09800(v94, (__int64 *)&v100, &v104);
          v37 = (unsigned __int64 *)v104;
          v25 = v100;
          v38 = *(_DWORD *)(v23 + 56) + 1;
        }
        else if ( v36 - *(_DWORD *)(v23 + 60) - v38 <= v36 >> 3 )
        {
          goto LABEL_32;
        }
        *(_DWORD *)(v23 + 56) = v38;
        if ( *v37 != -8 )
          --*(_DWORD *)(v23 + 60);
        *v37 = v25;
        v30 = 0;
        v37[1] = 0;
      }
LABEL_19:
      v101 = v30;
      v31 = *(_QWORD **)(v24 + 16);
      v32 = *(_QWORD **)(v24 + 8);
      if ( v31 == v32 )
      {
        v33 = &v32[*(unsigned int *)(v24 + 28)];
        if ( v32 == v33 )
        {
          v88 = *(_QWORD **)(v24 + 8);
        }
        else
        {
          do
          {
            if ( v35 == *v32 )
              break;
            ++v32;
          }
          while ( v33 != v32 );
          v88 = v33;
        }
      }
      else
      {
        v33 = &v31[*(unsigned int *)(v24 + 24)];
        v32 = sub_16CC9F0(v24, v35);
        if ( v35 == *v32 )
        {
          v71 = *(_QWORD *)(v24 + 16);
          if ( v71 == *(_QWORD *)(v24 + 8) )
            v72 = *(unsigned int *)(v24 + 28);
          else
            v72 = *(unsigned int *)(v24 + 24);
          v88 = (_QWORD *)(v71 + 8 * v72);
        }
        else
        {
          v34 = *(_QWORD *)(v24 + 16);
          if ( v34 != *(_QWORD *)(v24 + 8) )
          {
            v32 = (_QWORD *)(v34 + 8LL * *(unsigned int *)(v24 + 24));
            goto LABEL_23;
          }
          v32 = (_QWORD *)(v34 + 8LL * *(unsigned int *)(v24 + 28));
          v88 = v32;
        }
      }
      while ( v88 != v32 && *v32 >= 0xFFFFFFFFFFFFFFFELL )
        ++v32;
LABEL_23:
      if ( v33 != v32 )
      {
        *(_BYTE *)(v101 + 8) = 1;
        goto LABEL_25;
      }
      v39 = *(_QWORD **)(a3 + 24);
      if ( v39 )
      {
        v40 = (_QWORD *)(a3 + 16);
        do
        {
          while ( 1 )
          {
            v41 = v39[2];
            v42 = v39[3];
            if ( v39[4] >= v100 )
              break;
            v39 = (_QWORD *)v39[3];
            if ( !v42 )
              goto LABEL_38;
          }
          v40 = v39;
          v39 = (_QWORD *)v39[2];
        }
        while ( v41 );
LABEL_38:
        if ( v40 != (_QWORD *)(a3 + 16) && v40[4] <= v100 )
        {
          v105 = 0;
          v106 = 0;
          v107 = &v105;
          v108 = &v105;
          v109 = 0;
          v43 = v40[7];
          if ( v43 )
          {
            v44 = sub_1C07830(v43, (__int64)&v105);
            v43 = v44;
            do
            {
              v45 = v44;
              v44 = *(_QWORD *)(v44 + 16);
            }
            while ( v44 );
            v107 = (int *)v45;
            v46 = v43;
            do
            {
              v47 = (int *)v46;
              v46 = *(_QWORD *)(v46 + 24);
            }
            while ( v46 );
            v108 = v47;
            v48 = v40[10];
            v106 = v43;
            v109 = v48;
            if ( (int *)v45 != &v105 )
            {
              v93 = v22;
              while ( 1 )
              {
                v102 = *(_QWORD *)(v45 + 32);
                v49 = sub_1C08A20(v23, v102, v100, a4);
                if ( v49 )
                {
                  v50 = v101;
                  v51 = sub_1C0A960((int *)v23, v102, v49);
                  v54 = *(unsigned int *)(v50 + 32);
                  v55 = v51;
                  if ( (_DWORD)v54 )
                  {
                    v56 = *(_QWORD **)(v50 + 24);
                    v57 = (__int64)&v56[(unsigned int)(v54 - 1) + 1];
                    while ( v55 != *v56 )
                    {
                      if ( (_QWORD *)v57 == ++v56 )
                        goto LABEL_81;
                    }
                    v58 = *(_DWORD *)(a7 + 24);
                    if ( !v58 )
                    {
LABEL_84:
                      ++*(_QWORD *)a7;
                      goto LABEL_85;
                    }
                  }
                  else
                  {
LABEL_81:
                    if ( (unsigned int)v54 >= *(_DWORD *)(v50 + 36) )
                    {
                      sub_16CD150(v50 + 24, (const void *)(v50 + 40), 0, 8, v52, v53);
                      v54 = *(unsigned int *)(v50 + 32);
                    }
                    *(_QWORD *)(*(_QWORD *)(v50 + 24) + 8 * v54) = v55;
                    ++*(_DWORD *)(v50 + 32);
                    v58 = *(_DWORD *)(a7 + 24);
                    if ( !v58 )
                      goto LABEL_84;
                  }
                  v59 = v102;
                  v60 = *(_QWORD *)(a7 + 8);
                  v61 = (v58 - 1) & (((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4));
                  v62 = v60 + 40LL * v61;
                  v63 = *(_QWORD *)v62;
                  if ( *(_QWORD *)v62 != v102 )
                  {
                    v76 = 0;
                    v84 = 1;
                    while ( v63 != -8 )
                    {
                      if ( !v76 && v63 == -16 )
                        v76 = v62;
                      v61 = (v58 - 1) & (v84 + v61);
                      v62 = v60 + 40LL * v61;
                      v63 = *(_QWORD *)v62;
                      if ( v102 == *(_QWORD *)v62 )
                        goto LABEL_54;
                      ++v84;
                    }
                    v85 = *(_DWORD *)(a7 + 16);
                    if ( !v76 )
                      v76 = v62;
                    ++*(_QWORD *)a7;
                    v77 = v85 + 1;
                    if ( 4 * (v85 + 1) >= 3 * v58 )
                    {
LABEL_85:
                      sub_1C0BC70(a7, 2 * v58);
                    }
                    else
                    {
                      if ( v58 - *(_DWORD *)(a7 + 20) - v77 > v58 >> 3 )
                      {
LABEL_112:
                        *(_DWORD *)(a7 + 16) = v77;
                        if ( *(_QWORD *)v76 != -8 )
                          --*(_DWORD *)(a7 + 20);
                        *(_QWORD *)v76 = v59;
                        v66 = v76 + 8;
                        v86 = 1;
                        *(_QWORD *)(v76 + 8) = 0;
                        *(_QWORD *)(v76 + 16) = 0;
                        *(_QWORD *)(v76 + 24) = 0;
                        *(_DWORD *)(v76 + 32) = 0;
LABEL_115:
                        *(_QWORD *)(v76 + 8) = v86;
                        v87 = 0;
                        goto LABEL_116;
                      }
                      sub_1C0BC70(a7, v58);
                    }
                    sub_1C09AC0(a7, &v102, &v103);
                    v76 = (__int64)v103;
                    v59 = v102;
                    v77 = *(_DWORD *)(a7 + 16) + 1;
                    goto LABEL_112;
                  }
LABEL_54:
                  v64 = *(_DWORD *)(v62 + 32);
                  v65 = *(_QWORD *)(v62 + 16);
                  v66 = v62 + 8;
                  if ( !v64 )
                  {
                    v76 = v62;
                    v86 = *(_QWORD *)(v62 + 8) + 1LL;
                    goto LABEL_115;
                  }
                  v67 = v101;
                  v68 = (v64 - 1) & (((unsigned int)v101 >> 9) ^ ((unsigned int)v101 >> 4));
                  v69 = (__int64 *)(v65 + 8LL * v68);
                  v70 = *v69;
                  if ( v101 != *v69 )
                  {
                    v80 = 0;
                    v81 = 1;
                    while ( v70 != -8 )
                    {
                      if ( v70 != -16 || v80 )
                        v69 = v80;
                      v68 = (v64 - 1) & (v81 + v68);
                      v91 = (__int64 *)(v65 + 8LL * v68);
                      v70 = *v91;
                      if ( v101 == *v91 )
                        goto LABEL_56;
                      ++v81;
                      v80 = v69;
                      v69 = (__int64 *)(v65 + 8LL * v68);
                    }
                    v82 = *(_DWORD *)(v62 + 24);
                    if ( !v80 )
                      v80 = v69;
                    ++*(_QWORD *)(v62 + 8);
                    v83 = v82 + 1;
                    if ( 4 * v83 < 3 * v64 )
                    {
                      if ( v64 - *(_DWORD *)(v62 + 28) - v83 <= v64 >> 3 )
                      {
                        sub_1C0C7C0(v62 + 8, v64);
                        sub_1C09B70(v62 + 8, &v101, &v103);
                        v80 = v103;
                        v67 = v101;
                        v83 = *(_DWORD *)(v62 + 24) + 1;
                      }
                      goto LABEL_103;
                    }
                    v87 = 2 * v64;
                    v76 = v62;
LABEL_116:
                    v90 = v76;
                    sub_1C0C7C0(v66, v87);
                    sub_1C09B70(v66, &v101, &v103);
                    v80 = v103;
                    v67 = v101;
                    v62 = v90;
                    v83 = *(_DWORD *)(v90 + 24) + 1;
LABEL_103:
                    *(_DWORD *)(v62 + 24) = v83;
                    if ( *v80 != -8 )
                      --*(_DWORD *)(v62 + 28);
                    *v80 = v67;
                  }
                }
LABEL_56:
                j___libc_free_0(0);
                v45 = sub_220EF30(v45);
                if ( (int *)v45 == &v105 )
                {
                  v22 = v93;
                  v43 = v106;
                  break;
                }
              }
            }
          }
          sub_1C07AA0(v43);
        }
      }
    }
LABEL_25:
    v22 = *(_QWORD *)(v22 + 8);
    if ( v98 != v22 )
      continue;
    break;
  }
  v8 = v23;
  return sub_1C0DCE0(v8, a2, a7);
}
