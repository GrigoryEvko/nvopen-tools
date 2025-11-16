// Function: sub_13EC960
// Address: 0x13ec960
//
__int64 __fastcall sub_13EC960(__int64 a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  unsigned int v7; // esi
  __int64 v8; // rdi
  unsigned int v9; // r14d
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rsi
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rbx
  char v20; // cl
  __int64 v21; // r8
  int v22; // esi
  unsigned int v23; // eax
  int *v24; // rdi
  __int64 v25; // rdx
  __int64 result; // rax
  unsigned int v27; // esi
  unsigned int v28; // esi
  __int64 v29; // rcx
  unsigned int v30; // edx
  __int64 *v31; // rdi
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 *v34; // r9
  _QWORD *v35; // rax
  _QWORD *v36; // r14
  _QWORD *v37; // rax
  unsigned int v38; // esi
  __int64 v39; // rcx
  __int64 v40; // r9
  unsigned int v41; // edx
  __int64 *v42; // rax
  __int64 v43; // r8
  __int64 v44; // r8
  __int64 v45; // rax
  __int64 v46; // r14
  __int64 v47; // r15
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdi
  unsigned int v51; // ecx
  __int64 v52; // r8
  int v53; // edx
  int v54; // r10d
  __int64 *v55; // rsi
  unsigned int v56; // r10d
  _QWORD *v57; // rcx
  unsigned int v58; // eax
  int v59; // edx
  unsigned int v60; // r8d
  __int64 v61; // rax
  int v62; // r10d
  __int64 *v63; // r9
  int v64; // eax
  int v65; // edx
  int v66; // r10d
  __int64 *v67; // r9
  int v68; // edx
  int v69; // r9d
  int v70; // eax
  int v71; // ecx
  __int64 v72; // rdi
  unsigned int v73; // eax
  __int64 v74; // rsi
  int v75; // r10d
  __int64 *v76; // r8
  int v77; // eax
  int v78; // eax
  __int64 v79; // rsi
  int v80; // r8d
  unsigned int v81; // r14d
  __int64 *v82; // rdi
  __int64 v83; // rcx
  __int64 v84; // rdi
  __int64 v85; // rdi
  int v86; // r11d
  __int64 *v87; // r10
  int v88; // eax
  int v89; // edx
  int v90; // eax
  int v91; // ecx
  __int64 v92; // r9
  __int64 v93; // rsi
  int v94; // r10d
  __int64 *v95; // r8
  int v96; // eax
  __int64 v97; // r9
  int v98; // r8d
  unsigned int v99; // r13d
  __int64 *v100; // rsi
  __int64 v101; // rcx
  int v102; // r11d
  __int64 *v103; // rdi
  int v104; // eax
  int v105; // edx
  __int64 v106; // [rsp+0h] [rbp-50h]
  __int64 v107; // [rsp+0h] [rbp-50h]
  __int64 v108; // [rsp+0h] [rbp-50h]
  __int64 v109; // [rsp+0h] [rbp-50h]
  __int64 v110; // [rsp+8h] [rbp-48h] BYREF
  __int64 v111; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v112[7]; // [rsp+18h] [rbp-38h] BYREF

  v110 = a2;
  v7 = *(_DWORD *)(a1 + 24);
  if ( !v7 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_91;
  }
  v8 = *(_QWORD *)(a1 + 8);
  v9 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v10 = (v7 - 1) & v9;
  v11 = (__int64 *)(v8 + 8LL * v10);
  v12 = *v11;
  if ( a3 == *v11 )
    goto LABEL_3;
  v62 = 1;
  v63 = 0;
  while ( v12 != -8 )
  {
    if ( v63 || v12 != -16 )
      v11 = v63;
    v10 = (v7 - 1) & (v62 + v10);
    v12 = *(_QWORD *)(v8 + 8LL * v10);
    if ( a3 == v12 )
      goto LABEL_3;
    ++v62;
    v63 = v11;
    v11 = (__int64 *)(v8 + 8LL * v10);
  }
  v64 = *(_DWORD *)(a1 + 16);
  if ( !v63 )
    v63 = v11;
  ++*(_QWORD *)a1;
  v65 = v64 + 1;
  if ( 4 * (v64 + 1) >= 3 * v7 )
  {
LABEL_91:
    sub_13EC060(a1, 2 * v7);
    v70 = *(_DWORD *)(a1 + 24);
    if ( v70 )
    {
      v71 = v70 - 1;
      v72 = *(_QWORD *)(a1 + 8);
      v73 = (v70 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v63 = (__int64 *)(v72 + 8LL * v73);
      v74 = *v63;
      v65 = *(_DWORD *)(a1 + 16) + 1;
      if ( a3 != *v63 )
      {
        v75 = 1;
        v76 = 0;
        while ( v74 != -8 )
        {
          if ( v74 == -16 && !v76 )
            v76 = v63;
          v73 = v71 & (v75 + v73);
          v63 = (__int64 *)(v72 + 8LL * v73);
          v74 = *v63;
          if ( a3 == *v63 )
            goto LABEL_68;
          ++v75;
        }
        if ( v76 )
          v63 = v76;
      }
      goto LABEL_68;
    }
    goto LABEL_184;
  }
  if ( v7 - *(_DWORD *)(a1 + 20) - v65 <= v7 >> 3 )
  {
    sub_13EC060(a1, v7);
    v77 = *(_DWORD *)(a1 + 24);
    if ( v77 )
    {
      v78 = v77 - 1;
      v79 = *(_QWORD *)(a1 + 8);
      v80 = 1;
      v81 = v78 & v9;
      v63 = (__int64 *)(v79 + 8LL * v81);
      v65 = *(_DWORD *)(a1 + 16) + 1;
      v82 = 0;
      v83 = *v63;
      if ( a3 != *v63 )
      {
        while ( v83 != -8 )
        {
          if ( v83 == -16 && !v82 )
            v82 = v63;
          v81 = v78 & (v80 + v81);
          v63 = (__int64 *)(v79 + 8LL * v81);
          v83 = *v63;
          if ( a3 == *v63 )
            goto LABEL_68;
          ++v80;
        }
        if ( v82 )
          v63 = v82;
      }
      goto LABEL_68;
    }
LABEL_184:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_68:
  *(_DWORD *)(a1 + 16) = v65;
  if ( *v63 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v63 = a3;
LABEL_3:
  if ( *a4 != 4 )
  {
    v13 = *(unsigned int *)(a1 + 56);
    v14 = v110;
    if ( (_DWORD)v13 )
    {
      v15 = *(_QWORD *)(a1 + 40);
      v16 = (v13 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
      v17 = (__int64 *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( *v17 == v110 )
      {
LABEL_6:
        if ( v17 != (__int64 *)(v15 + 16 * v13) )
          goto LABEL_7;
      }
      else
      {
        v68 = 1;
        while ( v18 != -8 )
        {
          v69 = v68 + 1;
          v16 = (v13 - 1) & (v68 + v16);
          v17 = (__int64 *)(v15 + 16LL * v16);
          v18 = *v17;
          if ( *v17 == v110 )
            goto LABEL_6;
          v68 = v69;
        }
      }
    }
    v35 = (_QWORD *)sub_22077B0(248);
    v36 = v35;
    if ( v35 )
    {
      v35[1] = 2;
      v35[2] = 0;
      v35[3] = v14;
      if ( v14 != 0 && v14 != -8 && v14 != -16 )
        sub_164C220(v35 + 1);
      v36[4] = a1;
      v36[5] = 0;
      v36[6] = 1;
      *v36 = off_49EA8F0;
      v37 = v36 + 7;
      do
      {
        if ( v37 )
          *v37 = -8;
        v37 += 6;
      }
      while ( v37 != v36 + 31 );
    }
    v38 = *(_DWORD *)(a1 + 56);
    if ( v38 )
    {
      v39 = v110;
      v40 = *(_QWORD *)(a1 + 40);
      v41 = (v38 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
      v42 = (__int64 *)(v40 + 16LL * v41);
      v43 = *v42;
      if ( v110 == *v42 )
      {
LABEL_28:
        v44 = v42[1];
        v42[1] = (__int64)v36;
        if ( !v44 )
          goto LABEL_41;
        if ( (*(_BYTE *)(v44 + 48) & 1) != 0 )
        {
          v46 = v44 + 56;
          v47 = v44 + 248;
        }
        else
        {
          v45 = *(unsigned int *)(v44 + 64);
          v46 = *(_QWORD *)(v44 + 56);
          if ( !(_DWORD)v45 )
            goto LABEL_99;
          v47 = v46 + 48 * v45;
        }
        do
        {
          if ( *(_QWORD *)v46 != -16 && *(_QWORD *)v46 != -8 && *(_DWORD *)(v46 + 8) == 3 )
          {
            if ( *(_DWORD *)(v46 + 40) > 0x40u )
            {
              v84 = *(_QWORD *)(v46 + 32);
              if ( v84 )
              {
                v108 = v44;
                j_j___libc_free_0_0(v84);
                v44 = v108;
              }
            }
            if ( *(_DWORD *)(v46 + 24) > 0x40u )
            {
              v85 = *(_QWORD *)(v46 + 16);
              if ( v85 )
              {
                v109 = v44;
                j_j___libc_free_0_0(v85);
                v44 = v109;
              }
            }
          }
          v46 += 48;
        }
        while ( v46 != v47 );
        if ( (*(_BYTE *)(v44 + 48) & 1) != 0 )
          goto LABEL_37;
        v46 = *(_QWORD *)(v44 + 56);
LABEL_99:
        v107 = v44;
        j___libc_free_0(v46);
        v44 = v107;
LABEL_37:
        *(_QWORD *)v44 = &unk_49EE2B0;
        v48 = *(_QWORD *)(v44 + 24);
        if ( v48 != -8 && v48 != 0 && v48 != -16 )
        {
          v106 = v44;
          sub_1649B30(v44 + 8);
          v44 = v106;
        }
        j_j___libc_free_0(v44, 248);
        goto LABEL_41;
      }
      v102 = 1;
      v103 = 0;
      while ( v43 != -8 )
      {
        if ( !v103 && v43 == -16 )
          v103 = v42;
        v41 = (v38 - 1) & (v102 + v41);
        v42 = (__int64 *)(v40 + 16LL * v41);
        v43 = *v42;
        if ( v110 == *v42 )
          goto LABEL_28;
        ++v102;
      }
      if ( !v103 )
        v103 = v42;
      v104 = *(_DWORD *)(a1 + 48);
      ++*(_QWORD *)(a1 + 32);
      v105 = v104 + 1;
      if ( 4 * (v104 + 1) < 3 * v38 )
      {
        if ( v38 - *(_DWORD *)(a1 + 52) - v105 > v38 >> 3 )
        {
LABEL_143:
          *(_DWORD *)(a1 + 48) = v105;
          if ( *v103 != -8 )
            --*(_DWORD *)(a1 + 52);
          *v103 = v39;
          v103[1] = (__int64)v36;
LABEL_41:
          v49 = *(unsigned int *)(a1 + 56);
          v50 = *(_QWORD *)(a1 + 40);
          if ( (_DWORD)v49 )
          {
            v51 = (v49 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
            v17 = (__int64 *)(v50 + 16LL * v51);
            v52 = *v17;
            if ( v110 == *v17 )
              goto LABEL_7;
            v53 = 1;
            while ( v52 != -8 )
            {
              v54 = v53 + 1;
              v51 = (v49 - 1) & (v53 + v51);
              v17 = (__int64 *)(v50 + 16LL * v51);
              v52 = *v17;
              if ( v110 == *v17 )
                goto LABEL_7;
              v53 = v54;
            }
          }
          v17 = (__int64 *)(v50 + 16 * v49);
LABEL_7:
          v19 = v17[1];
          v111 = a3;
          v20 = *(_BYTE *)(v19 + 48) & 1;
          if ( v20 )
          {
            v21 = v19 + 56;
            v22 = 3;
          }
          else
          {
            v27 = *(_DWORD *)(v19 + 64);
            v21 = *(_QWORD *)(v19 + 56);
            if ( !v27 )
            {
              v58 = *(_DWORD *)(v19 + 48);
              ++*(_QWORD *)(v19 + 40);
              v24 = 0;
              v59 = (v58 >> 1) + 1;
              goto LABEL_56;
            }
            v22 = v27 - 1;
          }
          v23 = v22 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
          v24 = (int *)(v21 + 48LL * v23);
          v25 = *(_QWORD *)v24;
          if ( a3 == *(_QWORD *)v24 )
            return sub_13E8810(v24 + 2, a4);
          v66 = 1;
          v67 = 0;
          while ( v25 != -8 )
          {
            if ( !v67 && v25 == -16 )
              v67 = (__int64 *)v24;
            v23 = v22 & (v66 + v23);
            v24 = (int *)(v21 + 48LL * v23);
            v25 = *(_QWORD *)v24;
            if ( a3 == *(_QWORD *)v24 )
              return sub_13E8810(v24 + 2, a4);
            ++v66;
          }
          v58 = *(_DWORD *)(v19 + 48);
          v60 = 12;
          v27 = 4;
          if ( v67 )
            v24 = (int *)v67;
          ++*(_QWORD *)(v19 + 40);
          v59 = (v58 >> 1) + 1;
          if ( v20 )
          {
LABEL_57:
            if ( v60 <= 4 * v59 )
            {
              v27 *= 2;
            }
            else if ( v27 - *(_DWORD *)(v19 + 52) - v59 > v27 >> 3 )
            {
LABEL_59:
              *(_DWORD *)(v19 + 48) = (2 * (v58 >> 1) + 2) | v58 & 1;
              if ( *(_QWORD *)v24 != -8 )
                --*(_DWORD *)(v19 + 52);
              v61 = v111;
              v24[2] = 0;
              *(_QWORD *)v24 = v61;
              return sub_13E8810(v24 + 2, a4);
            }
            sub_13EC410(v19 + 40, v27);
            sub_13EBE70(v19 + 40, &v111, v112);
            v24 = (int *)v112[0];
            v58 = *(_DWORD *)(v19 + 48);
            goto LABEL_59;
          }
          v27 = *(_DWORD *)(v19 + 64);
LABEL_56:
          v60 = 3 * v27;
          goto LABEL_57;
        }
LABEL_148:
        sub_13E81B0(a1 + 32, v38);
        sub_13E7480(a1 + 32, &v110, v112);
        v103 = (__int64 *)v112[0];
        v39 = v110;
        v105 = *(_DWORD *)(a1 + 48) + 1;
        goto LABEL_143;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 32);
    }
    v38 *= 2;
    goto LABEL_148;
  }
  v28 = *(_DWORD *)(a1 + 88);
  if ( !v28 )
  {
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_124;
  }
  v29 = *(_QWORD *)(a1 + 72);
  v30 = (v28 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v31 = (__int64 *)(v29 + 80LL * v30);
  v32 = *v31;
  if ( a3 != *v31 )
  {
    v86 = 1;
    v87 = 0;
    while ( v32 != -8 )
    {
      if ( v32 == -16 && !v87 )
        v87 = v31;
      v30 = (v28 - 1) & (v86 + v30);
      v31 = (__int64 *)(v29 + 80LL * v30);
      v32 = *v31;
      if ( a3 == *v31 )
        goto LABEL_16;
      ++v86;
    }
    v88 = *(_DWORD *)(a1 + 80);
    if ( v87 )
      v31 = v87;
    ++*(_QWORD *)(a1 + 64);
    v89 = v88 + 1;
    if ( 4 * (v88 + 1) < 3 * v28 )
    {
      result = v28 - *(_DWORD *)(a1 + 84) - v89;
      if ( (unsigned int)result > v28 >> 3 )
      {
LABEL_118:
        *(_DWORD *)(a1 + 80) = v89;
        if ( *v31 != -8 )
          --*(_DWORD *)(a1 + 84);
        v55 = v31 + 6;
        *v31 = a3;
        v33 = v110;
        v34 = v31 + 1;
        v31[1] = 0;
        v56 = 0;
        v31[2] = (__int64)(v31 + 6);
        v31[3] = (__int64)(v31 + 6);
        v31[4] = 4;
        *((_DWORD *)v31 + 10) = 0;
        goto LABEL_121;
      }
      sub_13EC210(a1 + 64, v28);
      v96 = *(_DWORD *)(a1 + 88);
      if ( v96 )
      {
        result = (unsigned int)(v96 - 1);
        v97 = *(_QWORD *)(a1 + 72);
        v98 = 1;
        v99 = result & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v89 = *(_DWORD *)(a1 + 80) + 1;
        v100 = 0;
        v31 = (__int64 *)(v97 + 80LL * v99);
        v101 = *v31;
        if ( a3 != *v31 )
        {
          while ( v101 != -8 )
          {
            if ( !v100 && v101 == -16 )
              v100 = v31;
            v99 = result & (v98 + v99);
            v31 = (__int64 *)(v97 + 80LL * v99);
            v101 = *v31;
            if ( a3 == *v31 )
              goto LABEL_118;
            ++v98;
          }
          if ( v100 )
            v31 = v100;
        }
        goto LABEL_118;
      }
LABEL_185:
      ++*(_DWORD *)(a1 + 80);
      BUG();
    }
LABEL_124:
    sub_13EC210(a1 + 64, 2 * v28);
    v90 = *(_DWORD *)(a1 + 88);
    if ( v90 )
    {
      v91 = v90 - 1;
      v92 = *(_QWORD *)(a1 + 72);
      result = (v90 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v31 = (__int64 *)(v92 + 80 * result);
      v89 = *(_DWORD *)(a1 + 80) + 1;
      v93 = *v31;
      if ( a3 != *v31 )
      {
        v94 = 1;
        v95 = 0;
        while ( v93 != -8 )
        {
          if ( v93 == -16 && !v95 )
            v95 = v31;
          result = v91 & (unsigned int)(v94 + result);
          v31 = (__int64 *)(v92 + 80LL * (unsigned int)result);
          v93 = *v31;
          if ( a3 == *v31 )
            goto LABEL_118;
          ++v94;
        }
        if ( v95 )
          v31 = v95;
      }
      goto LABEL_118;
    }
    goto LABEL_185;
  }
LABEL_16:
  v33 = v110;
  result = v31[2];
  v34 = v31 + 1;
  if ( v31[3] != result )
    return sub_16CCBA0(v34, v33);
  v55 = (__int64 *)(result + 8LL * *((unsigned int *)v31 + 9));
  v56 = *((_DWORD *)v31 + 9);
  if ( (__int64 *)result == v55 )
  {
LABEL_121:
    if ( *((_DWORD *)v31 + 8) > v56 )
    {
      *((_DWORD *)v31 + 9) = v56 + 1;
      *v55 = v33;
      ++v31[1];
      return result;
    }
    return sub_16CCBA0(v34, v33);
  }
  v57 = 0;
  while ( v110 != *(_QWORD *)result )
  {
    if ( *(_QWORD *)result == -2 )
      v57 = (_QWORD *)result;
    result += 8;
    if ( (__int64 *)result == v55 )
    {
      if ( !v57 )
        goto LABEL_121;
      *v57 = v110;
      --*((_DWORD *)v31 + 10);
      ++v31[1];
      return result;
    }
  }
  return result;
}
