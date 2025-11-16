// Function: sub_3742B00
// Address: 0x3742b00
//
unsigned int *__fastcall sub_3742B00(__int64 a1, _BYTE *a2, unsigned int a3, int a4)
{
  __int64 v4; // rax
  __int64 v6; // rbx
  unsigned int v7; // esi
  __int64 v8; // r8
  int v9; // r10d
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // rdi
  unsigned int *result; // rax
  int v14; // r12d
  unsigned int *v15; // rdx
  unsigned int v16; // r14d
  int v17; // r15d
  __int64 v18; // rax
  int v19; // r13d
  __int64 v20; // r11
  __int64 v21; // r9
  _DWORD *v22; // rax
  int v23; // ecx
  unsigned int *v24; // rax
  __int64 v25; // rbx
  unsigned int v26; // esi
  __int64 v27; // rdi
  unsigned int v28; // ecx
  unsigned int *v29; // rdx
  unsigned int v30; // eax
  unsigned int v31; // esi
  int v32; // r12d
  int v33; // eax
  int v34; // eax
  __int64 v35; // rdi
  unsigned int v36; // r9d
  int v37; // ecx
  _DWORD *v38; // rdx
  int v39; // esi
  unsigned int v40; // esi
  __int64 v41; // r12
  __int64 v42; // rdi
  int v43; // ebx
  __int64 *v44; // r9
  unsigned int v45; // ecx
  _QWORD *v46; // rdx
  __int64 v47; // r10
  int v48; // edx
  int v49; // esi
  __int64 v50; // r8
  unsigned int v51; // ecx
  __int64 v52; // rdi
  int v53; // edi
  int v54; // r12d
  unsigned int *v55; // r10
  int v56; // eax
  int v57; // eax
  int v58; // eax
  int v59; // eax
  int v60; // eax
  _DWORD *v61; // r10
  __int64 v62; // rdi
  int v63; // r8d
  unsigned int v64; // r9d
  int v65; // esi
  int v66; // edx
  unsigned int v67; // edx
  __int64 v68; // rdi
  __int64 v69; // rsi
  unsigned int v70; // ecx
  int v71; // r12d
  unsigned int *v72; // r9
  int v73; // edx
  unsigned int v74; // edx
  int v75; // r12d
  __int64 v76; // rdi
  __int64 v77; // rcx
  unsigned int v78; // esi
  int v79; // edi
  int v80; // edi
  int v81; // ecx
  int v82; // r8d
  int v83; // r10d
  __int64 *v84; // r9
  int v85; // [rsp+8h] [rbp-68h]
  __int64 v86; // [rsp+8h] [rbp-68h]
  unsigned int v87; // [rsp+10h] [rbp-60h]
  __int64 v88; // [rsp+10h] [rbp-60h]
  int v89; // [rsp+10h] [rbp-60h]
  __int64 v90; // [rsp+10h] [rbp-60h]
  __int64 v91; // [rsp+10h] [rbp-60h]
  __int64 v92; // [rsp+18h] [rbp-58h]
  __int64 v95[2]; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v96[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = (__int64)a2;
  v95[0] = (__int64)a2;
  if ( *a2 <= 0x1Cu )
  {
    v40 = *(_DWORD *)(a1 + 32);
    v41 = a1 + 8;
    if ( v40 )
    {
      v42 = *(_QWORD *)(a1 + 16);
      v43 = 1;
      v44 = 0;
      v45 = (v40 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v46 = (_QWORD *)(v42 + 16LL * v45);
      v47 = *v46;
      if ( v4 == *v46 )
      {
LABEL_25:
        result = (unsigned int *)(v46 + 1);
LABEL_26:
        *result = a3;
        return result;
      }
      while ( v47 != -4096 )
      {
        if ( !v44 && v47 == -8192 )
          v44 = v46;
        v45 = (v40 - 1) & (v43 + v45);
        v46 = (_QWORD *)(v42 + 16LL * v45);
        v47 = *v46;
        if ( v4 == *v46 )
          goto LABEL_25;
        ++v43;
      }
      v80 = *(_DWORD *)(a1 + 24);
      if ( !v44 )
        v44 = v46;
      ++*(_QWORD *)(a1 + 8);
      v81 = v80 + 1;
      v96[0] = v44;
      if ( 4 * (v80 + 1) < 3 * v40 )
      {
        if ( v40 - *(_DWORD *)(a1 + 28) - v81 > v40 >> 3 )
        {
LABEL_93:
          *(_DWORD *)(a1 + 24) = v81;
          if ( *v44 != -4096 )
            --*(_DWORD *)(a1 + 28);
          *v44 = v4;
          result = (unsigned int *)(v44 + 1);
          *((_DWORD *)v44 + 2) = 0;
          goto LABEL_26;
        }
LABEL_98:
        sub_3384500(v41, v40);
        sub_337AD60(v41, v95, v96);
        v4 = v95[0];
        v44 = (__int64 *)v96[0];
        v81 = *(_DWORD *)(a1 + 24) + 1;
        goto LABEL_93;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 8);
      v96[0] = 0;
    }
    v40 *= 2;
    goto LABEL_98;
  }
  v6 = *(_QWORD *)(a1 + 40);
  v7 = *(_DWORD *)(v6 + 144);
  if ( !v7 )
  {
    v96[0] = 0;
    ++*(_QWORD *)(v6 + 120);
    goto LABEL_28;
  }
  v8 = *(_QWORD *)(v6 + 128);
  v9 = 1;
  v10 = (v7 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v11 = 0;
  v92 = v8 + 16LL * v10;
  v12 = *(_QWORD *)v92;
  if ( v4 != *(_QWORD *)v92 )
  {
    while ( v12 != -4096 )
    {
      if ( !v11 && v12 == -8192 )
        v11 = (__int64 *)v92;
      v10 = (v7 - 1) & (v9 + v10);
      v92 = v8 + 16LL * v10;
      v12 = *(_QWORD *)v92;
      if ( v4 == *(_QWORD *)v92 )
        goto LABEL_4;
      ++v9;
    }
    if ( !v11 )
      v11 = (__int64 *)v92;
    v96[0] = v11;
    v79 = *(_DWORD *)(v6 + 136);
    ++*(_QWORD *)(v6 + 120);
    v53 = v79 + 1;
    if ( 4 * v53 < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(v6 + 140) - v53 <= v7 >> 3 )
      {
        sub_3384500(v6 + 120, v7);
        sub_337AD60(v6 + 120, v95, v96);
        v4 = v95[0];
        v11 = (__int64 *)v96[0];
        v53 = *(_DWORD *)(v6 + 136) + 1;
      }
LABEL_31:
      *(_DWORD *)(v6 + 136) = v53;
      if ( *v11 != -4096 )
        --*(_DWORD *)(v6 + 140);
      *v11 = v4;
      v15 = (unsigned int *)(v11 + 1);
      *v15 = 0;
      goto LABEL_34;
    }
LABEL_28:
    sub_3384500(v6 + 120, 2 * v7);
    v48 = *(_DWORD *)(v6 + 144);
    if ( v48 )
    {
      v4 = v95[0];
      v49 = v48 - 1;
      v50 = *(_QWORD *)(v6 + 128);
      v51 = (v48 - 1) & ((LODWORD(v95[0]) >> 9) ^ (LODWORD(v95[0]) >> 4));
      v11 = (__int64 *)(v50 + 16LL * v51);
      v52 = *v11;
      if ( v95[0] == *v11 )
      {
LABEL_30:
        v96[0] = v11;
        v53 = *(_DWORD *)(v6 + 136) + 1;
      }
      else
      {
        v83 = 1;
        v84 = 0;
        while ( v52 != -4096 )
        {
          if ( v52 == -8192 && !v84 )
            v84 = v11;
          v51 = v49 & (v83 + v51);
          v11 = (__int64 *)(v50 + 16LL * v51);
          v52 = *v11;
          if ( v95[0] == *v11 )
            goto LABEL_30;
          ++v83;
        }
        if ( !v84 )
          v84 = v11;
        v96[0] = v84;
        v11 = v84;
        v53 = *(_DWORD *)(v6 + 136) + 1;
      }
    }
    else
    {
      v96[0] = 0;
      v11 = 0;
      v4 = v95[0];
      v53 = *(_DWORD *)(v6 + 136) + 1;
    }
    goto LABEL_31;
  }
LABEL_4:
  result = (unsigned int *)v92;
  v14 = *(_DWORD *)(v92 + 8);
  v15 = (unsigned int *)(v92 + 8);
  if ( !v14 )
  {
LABEL_34:
    *v15 = a3;
    return (unsigned int *)a3;
  }
  v16 = a3;
  if ( a3 != v14 )
  {
    if ( a4 )
    {
      v17 = 37 * a3;
      v18 = a1;
      v19 = 0;
      v20 = v18;
      while ( 1 )
      {
        v31 = *(_DWORD *)(v6 + 488);
        v32 = v19 + v14;
        if ( v31 )
        {
          v21 = *(_QWORD *)(v6 + 472);
          v87 = (v31 - 1) & (37 * v32);
          v22 = (_DWORD *)(v21 + 8LL * v87);
          v23 = *v22;
          if ( v32 == *v22 )
          {
LABEL_9:
            v24 = v22 + 1;
            goto LABEL_10;
          }
          v85 = 1;
          v38 = 0;
          while ( v23 != -1 )
          {
            if ( !v38 && v23 == -2 )
              v38 = v22;
            v87 = (v31 - 1) & (v87 + v85);
            v22 = (_DWORD *)(v21 + 8LL * v87);
            v23 = *v22;
            if ( v32 == *v22 )
              goto LABEL_9;
            ++v85;
          }
          if ( !v38 )
            v38 = v22;
          v58 = *(_DWORD *)(v6 + 480);
          ++*(_QWORD *)(v6 + 464);
          v37 = v58 + 1;
          if ( 4 * (v58 + 1) < 3 * v31 )
          {
            if ( v31 - *(_DWORD *)(v6 + 484) - v37 > v31 >> 3 )
              goto LABEL_18;
            v86 = v20;
            v89 = 37 * v32;
            sub_2FFACA0(v6 + 464, v31);
            v59 = *(_DWORD *)(v6 + 488);
            if ( !v59 )
            {
LABEL_137:
              ++*(_DWORD *)(v6 + 480);
              BUG();
            }
            v60 = v59 - 1;
            v61 = 0;
            v62 = *(_QWORD *)(v6 + 472);
            v20 = v86;
            v37 = *(_DWORD *)(v6 + 480) + 1;
            v63 = 1;
            v64 = v60 & v89;
            v38 = (_DWORD *)(v62 + 8LL * (v60 & (unsigned int)v89));
            v65 = *v38;
            if ( v32 == *v38 )
              goto LABEL_18;
            while ( v65 != -1 )
            {
              if ( !v61 && v65 == -2 )
                v61 = v38;
              v64 = v60 & (v63 + v64);
              v38 = (_DWORD *)(v62 + 8LL * v64);
              v65 = *v38;
              if ( v32 == *v38 )
                goto LABEL_18;
              ++v63;
            }
            goto LABEL_53;
          }
        }
        else
        {
          ++*(_QWORD *)(v6 + 464);
        }
        v88 = v20;
        sub_2FFACA0(v6 + 464, 2 * v31);
        v33 = *(_DWORD *)(v6 + 488);
        if ( !v33 )
          goto LABEL_137;
        v34 = v33 - 1;
        v35 = *(_QWORD *)(v6 + 472);
        v20 = v88;
        v36 = v34 & (37 * v32);
        v37 = *(_DWORD *)(v6 + 480) + 1;
        v38 = (_DWORD *)(v35 + 8LL * v36);
        v39 = *v38;
        if ( v32 == *v38 )
          goto LABEL_18;
        v82 = 1;
        v61 = 0;
        while ( v39 != -1 )
        {
          if ( v39 == -2 && !v61 )
            v61 = v38;
          v36 = v34 & (v82 + v36);
          v38 = (_DWORD *)(v35 + 8LL * v36);
          v39 = *v38;
          if ( v32 == *v38 )
            goto LABEL_18;
          ++v82;
        }
LABEL_53:
        if ( v61 )
          v38 = v61;
LABEL_18:
        *(_DWORD *)(v6 + 480) = v37;
        if ( *v38 != -1 )
          --*(_DWORD *)(v6 + 484);
        *v38 = v32;
        v24 = v38 + 1;
        v38[1] = 0;
LABEL_10:
        *v24 = v16;
        v25 = *(_QWORD *)(v20 + 40);
        v26 = *(_DWORD *)(v25 + 520);
        if ( v26 )
        {
          v27 = *(_QWORD *)(v25 + 504);
          v28 = v17 & (v26 - 1);
          v29 = (unsigned int *)(v27 + 4LL * v28);
          v30 = *v29;
          if ( *v29 == v16 )
            goto LABEL_12;
          v54 = 1;
          v55 = 0;
          while ( v30 != -1 )
          {
            if ( v55 || v30 != -2 )
              v29 = v55;
            v28 = (v26 - 1) & (v54 + v28);
            v30 = *(_DWORD *)(v27 + 4LL * v28);
            if ( v30 == v16 )
              goto LABEL_12;
            ++v54;
            v55 = v29;
            v29 = (unsigned int *)(v27 + 4LL * v28);
          }
          v56 = *(_DWORD *)(v25 + 512);
          if ( !v55 )
            v55 = v29;
          ++*(_QWORD *)(v25 + 496);
          v57 = v56 + 1;
          if ( 4 * v57 < 3 * v26 )
          {
            if ( v26 - *(_DWORD *)(v25 + 516) - v57 > v26 >> 3 )
              goto LABEL_41;
            v91 = v20;
            sub_2E29BA0(v25 + 496, v26);
            v73 = *(_DWORD *)(v25 + 520);
            if ( !v73 )
            {
LABEL_138:
              ++*(_DWORD *)(v25 + 512);
              BUG();
            }
            v74 = v73 - 1;
            v72 = 0;
            v20 = v91;
            v75 = 1;
            v76 = *(_QWORD *)(v25 + 504);
            LODWORD(v77) = v17 & v74;
            v55 = (unsigned int *)(v76 + 4LL * (v17 & v74));
            v78 = *v55;
            v57 = *(_DWORD *)(v25 + 512) + 1;
            if ( *v55 == v16 )
              goto LABEL_41;
            while ( v78 != -1 )
            {
              if ( v78 == -2 && !v72 )
                v72 = v55;
              v77 = v74 & ((_DWORD)v77 + v75);
              v55 = (unsigned int *)(v76 + 4 * v77);
              v78 = *v55;
              if ( v16 == *v55 )
                goto LABEL_41;
              ++v75;
            }
            goto LABEL_61;
          }
        }
        else
        {
          ++*(_QWORD *)(v25 + 496);
        }
        v90 = v20;
        sub_2E29BA0(v25 + 496, 2 * v26);
        v66 = *(_DWORD *)(v25 + 520);
        if ( !v66 )
          goto LABEL_138;
        v67 = v66 - 1;
        v68 = *(_QWORD *)(v25 + 504);
        v20 = v90;
        LODWORD(v69) = v17 & v67;
        v55 = (unsigned int *)(v68 + 4LL * (v17 & v67));
        v70 = *v55;
        v57 = *(_DWORD *)(v25 + 512) + 1;
        if ( *v55 == v16 )
          goto LABEL_41;
        v71 = 1;
        v72 = 0;
        while ( v70 != -1 )
        {
          if ( !v72 && v70 == -2 )
            v72 = v55;
          v69 = v67 & ((_DWORD)v69 + v71);
          v55 = (unsigned int *)(v68 + 4 * v69);
          v70 = *v55;
          if ( *v55 == v16 )
            goto LABEL_41;
          ++v71;
        }
LABEL_61:
        if ( v72 )
          v55 = v72;
LABEL_41:
        *(_DWORD *)(v25 + 512) = v57;
        if ( *v55 != -1 )
          --*(_DWORD *)(v25 + 516);
        *v55 = v16;
LABEL_12:
        ++v19;
        ++v16;
        v17 += 37;
        if ( a4 == v19 )
          break;
        v6 = *(_QWORD *)(v20 + 40);
        v14 = *(_DWORD *)(v92 + 8);
      }
    }
    result = (unsigned int *)v92;
    *(_DWORD *)(v92 + 8) = a3;
  }
  return result;
}
