// Function: sub_1C4A920
// Address: 0x1c4a920
//
__int64 __fastcall sub_1C4A920(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // rdx
  __int64 result; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // r10d
  __int64 v10; // r9
  __int64 v11; // rdi
  unsigned int v12; // edx
  __int64 v13; // r11
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // rdx
  unsigned int v17; // r11d
  int v18; // ecx
  int v19; // ecx
  __int64 v20; // r10
  unsigned int v21; // r9d
  __int64 *v22; // rdi
  __int64 v23; // r8
  unsigned int v24; // r13d
  int v25; // eax
  __int64 **v26; // r13
  __int64 *v27; // rdi
  int v28; // r8d
  __int64 v29; // r9
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 v34; // r9
  const void *v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 **v38; // r12
  __int64 v39; // r14
  __int64 v40; // rbx
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 v43; // rax
  __int64 v44; // rdx
  int v45; // eax
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rdx
  int v49; // edi
  int v50; // r13d
  unsigned int v51; // esi
  _QWORD *v52; // rdi
  unsigned int v53; // eax
  __int64 v54; // rax
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rax
  int v57; // ebx
  __int64 v58; // r13
  __int64 v59; // rdx
  __int64 k; // rdx
  __int64 *v61; // rbx
  __int64 *v62; // r14
  __int64 v63; // r13
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 *v67; // rdx
  unsigned int v68; // esi
  __int64 v69; // r8
  unsigned int v70; // edi
  _QWORD *v71; // rcx
  __int64 v72; // rdx
  int v73; // r11d
  _QWORD *v74; // r10
  int v75; // edi
  int v76; // ecx
  int v77; // eax
  int v78; // esi
  __int64 v79; // rdi
  __int64 v80; // rax
  __int64 v81; // rdx
  int v82; // r9d
  _QWORD *v83; // r8
  int v84; // edx
  int v85; // edx
  __int64 v86; // r9
  _QWORD *v87; // rsi
  unsigned int v88; // eax
  int v89; // edi
  __int64 v90; // r8
  _QWORD *v91; // rax
  int v92; // ecx
  unsigned int v93; // eax
  __int64 v94; // rdx
  __int64 j; // rdx
  _QWORD *v96; // rdi
  unsigned int v97; // eax
  __int64 v98; // rax
  unsigned __int64 v99; // rax
  unsigned __int64 v100; // rax
  int v101; // ebx
  __int64 v102; // r13
  __int64 v103; // rdx
  __int64 i; // rdx
  __int64 v105; // [rsp+10h] [rbp-40h]
  unsigned int v106; // [rsp+10h] [rbp-40h]
  __int64 v107; // [rsp+18h] [rbp-38h]
  __int64 v108; // [rsp+18h] [rbp-38h]

  v3 = a1;
  v5 = *(_QWORD *)(a1 + 48);
  *(_BYTE *)(a1 + 40) = 1;
  result = *(unsigned int *)(v5 + 8);
  if ( (_DWORD)result )
  {
    v7 = 0;
    v8 = 8LL * (unsigned int)(result - 1);
    while ( 1 )
    {
      result = *(unsigned int *)(a2 + 24);
      if ( (_DWORD)result )
      {
        v9 = result - 1;
        v10 = *(_QWORD *)(a2 + 8);
        v11 = *(_QWORD *)(*(_QWORD *)v5 + v7);
        v12 = (result - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        result = v10 + 16LL * v12;
        v13 = *(_QWORD *)result;
        if ( v11 == *(_QWORD *)result )
        {
LABEL_6:
          *(_QWORD *)result = -16;
          --*(_DWORD *)(a2 + 16);
          ++*(_DWORD *)(a2 + 20);
        }
        else
        {
          result = 1;
          while ( v13 != -8 )
          {
            v24 = result + 1;
            v12 = v9 & (result + v12);
            result = v10 + 16LL * v12;
            v13 = *(_QWORD *)result;
            if ( v11 == *(_QWORD *)result )
              goto LABEL_6;
            result = v24;
          }
        }
      }
      if ( v8 == v7 )
        break;
      v5 = *(_QWORD *)(v3 + 48);
      v7 += 8;
    }
  }
  v14 = *(unsigned int *)(v3 + 64);
  if ( (_DWORD)v14 )
  {
    v15 = 8 * v14;
    v16 = 0;
    v17 = ((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4);
    do
    {
      result = *(_QWORD *)(*(_QWORD *)(v3 + 56) + v16);
      if ( result )
      {
        v18 = *(_DWORD *)(result + 32);
        if ( v18 )
        {
          v19 = v18 - 1;
          v20 = *(_QWORD *)(result + 16);
          v21 = v19 & v17;
          v22 = (__int64 *)(v20 + 8LL * (v19 & v17));
          v23 = *v22;
          if ( v3 == *v22 )
          {
LABEL_13:
            *v22 = -16;
            --*(_DWORD *)(result + 24);
            ++*(_DWORD *)(result + 28);
          }
          else
          {
            v49 = 1;
            while ( v23 != -8 )
            {
              v50 = v49 + 1;
              v21 = v19 & (v49 + v21);
              v22 = (__int64 *)(v20 + 8LL * v21);
              v23 = *v22;
              if ( v3 == *v22 )
                goto LABEL_13;
              v49 = v50;
            }
          }
        }
        *(_BYTE *)(result + 40) = 1;
        result = *(_QWORD *)(v3 + 56);
        *(_QWORD *)(result + v16) = 0;
      }
      v16 += 8;
    }
    while ( v15 != v16 );
  }
  if ( *(_DWORD *)(v3 + 24) )
  {
    v25 = *(_DWORD *)(a3 + 8);
    v26 = *(__int64 ***)(v3 + 48);
    if ( v25 )
    {
      v27 = *(__int64 **)a3;
      v28 = *((_DWORD *)v26 + 2);
      v29 = *(_QWORD *)a3 + 8LL * (unsigned int)(v25 - 1) + 8;
      do
      {
        v30 = *v27;
        if ( v28 == *(_DWORD *)(*v27 + 56) )
        {
          if ( !v28 )
            goto LABEL_66;
          v31 = 0;
          while ( (*v26)[v31] == *(_QWORD *)(*(_QWORD *)(v30 + 48) + 8 * v31) )
          {
            if ( ++v31 == *((_DWORD *)v26 + 2) )
              goto LABEL_66;
          }
        }
        ++v27;
      }
      while ( (__int64 *)v29 != v27 );
    }
    v32 = sub_22077B0(88);
    v30 = v32;
    if ( v32 )
    {
      *(_QWORD *)(v32 + 8) = 0;
      v35 = (const void *)(v32 + 64);
      *(_QWORD *)(v32 + 16) = 0;
      *(_QWORD *)(v32 + 24) = 0;
      *(_DWORD *)(v32 + 32) = 0;
      *(_BYTE *)(v32 + 40) = 0;
      *(_QWORD *)v32 = off_4985478;
      *(_QWORD *)(v32 + 48) = v32 + 64;
      *(_QWORD *)(v32 + 56) = 0x200000000LL;
      v36 = *((unsigned int *)v26 + 2);
      if ( (_DWORD)v36 )
      {
        v34 = 8 * v36;
        v107 = v3;
        v37 = 0;
        v105 = a3;
        v38 = v26;
        v39 = 8;
        v40 = v34;
        v33 = **v26;
        v41 = v30 + 64;
        v42 = v33;
        while ( 1 )
        {
          *(_QWORD *)(v41 + 8 * v37) = v42;
          v37 = (unsigned int)(*(_DWORD *)(v30 + 56) + 1);
          *(_DWORD *)(v30 + 56) = v37;
          if ( v39 == v40 )
            break;
          v42 = (*v38)[(unsigned __int64)v39 / 8];
          if ( *(_DWORD *)(v30 + 60) <= (unsigned int)v37 )
          {
            sub_16CD150(v30 + 48, v35, 0, 8, v33, v34);
            v37 = *(unsigned int *)(v30 + 56);
          }
          v41 = *(_QWORD *)(v30 + 48);
          v39 += 8;
        }
        v3 = v107;
        a3 = v105;
      }
      *(_QWORD *)(v30 + 80) = 0;
    }
    v43 = *(unsigned int *)(a3 + 8);
    if ( (unsigned int)v43 >= *(_DWORD *)(a3 + 12) )
    {
      sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v33, v34);
      v43 = *(unsigned int *)(a3 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v43) = v30;
    ++*(_DWORD *)(a3 + 8);
    if ( !*(_DWORD *)(v3 + 24) )
    {
LABEL_42:
      v44 = *(_QWORD *)(v3 + 8);
      v45 = *(_DWORD *)(v3 + 24);
      v46 = v44 + 1;
      *(_QWORD *)(v3 + 8) = v44 + 1;
      if ( !v45 )
      {
        if ( *(_DWORD *)(v3 + 28) )
        {
          v47 = *(unsigned int *)(v3 + 32);
          if ( (unsigned int)v47 > 0x40 )
          {
            result = j___libc_free_0(*(_QWORD *)(v3 + 16));
            ++*(_QWORD *)(v3 + 8);
            *(_QWORD *)(v3 + 16) = 0;
            *(_QWORD *)(v3 + 24) = 0;
            *(_DWORD *)(v3 + 32) = 0;
            return result;
          }
          goto LABEL_45;
        }
        *(_QWORD *)(v3 + 8) = v44 + 2;
        goto LABEL_127;
      }
      v51 = 4 * v45;
      v47 = *(unsigned int *)(v3 + 32);
      if ( (unsigned int)(4 * v45) < 0x40 )
        v51 = 64;
      if ( v51 >= (unsigned int)v47 )
      {
LABEL_45:
        result = *(_QWORD *)(v3 + 16);
        v48 = result + 8 * v47;
        if ( result != v48 )
        {
          do
          {
            *(_QWORD *)result = -8;
            result += 8;
          }
          while ( v48 != result );
          v46 = *(_QWORD *)(v3 + 8);
        }
        *(_QWORD *)(v3 + 24) = 0;
        *(_QWORD *)(v3 + 8) = v46 + 1;
        return result;
      }
      v52 = *(_QWORD **)(v3 + 16);
      v53 = v45 - 1;
      if ( v53 )
      {
        _BitScanReverse(&v53, v53);
        v54 = (unsigned int)(1 << (33 - (v53 ^ 0x1F)));
        if ( (int)v54 < 64 )
          v54 = 64;
        if ( (_DWORD)v54 == (_DWORD)v47 )
        {
          *(_QWORD *)(v3 + 24) = 0;
          v91 = &v52[v54];
          do
          {
            if ( v52 )
              *v52 = -8;
            ++v52;
          }
          while ( v91 != v52 );
          v92 = *(_DWORD *)(v3 + 24);
          ++*(_QWORD *)(v3 + 8);
          if ( v92 )
          {
            v93 = 4 * v92;
            v94 = *(unsigned int *)(v3 + 32);
            if ( (unsigned int)(4 * v92) < 0x40 )
              v93 = 64;
            if ( v93 < (unsigned int)v94 )
            {
              v96 = *(_QWORD **)(v3 + 16);
              if ( v92 == 1 )
              {
                v102 = 1024;
                v101 = 128;
              }
              else
              {
                _BitScanReverse(&v97, v92 - 1);
                v98 = (unsigned int)(1 << (33 - (v97 ^ 0x1F)));
                if ( (int)v98 < 64 )
                  v98 = 64;
                if ( (_DWORD)v98 == (_DWORD)v94 )
                {
                  *(_QWORD *)(v3 + 24) = 0;
                  result = (__int64)&v96[v98];
                  do
                  {
                    if ( v96 )
                      *v96 = -8;
                    ++v96;
                  }
                  while ( (_QWORD *)result != v96 );
                  return result;
                }
                v99 = (4 * (int)v98 / 3u + 1) | ((unsigned __int64)(4 * (int)v98 / 3u + 1) >> 1);
                v100 = ((v99 | (v99 >> 2)) >> 4)
                     | v99
                     | (v99 >> 2)
                     | ((((v99 | (v99 >> 2)) >> 4) | v99 | (v99 >> 2)) >> 8);
                v101 = (v100 | (v100 >> 16)) + 1;
                v102 = 8 * ((v100 | (v100 >> 16)) + 1);
              }
              j___libc_free_0(v96);
              *(_DWORD *)(v3 + 32) = v101;
              result = sub_22077B0(v102);
              v103 = *(unsigned int *)(v3 + 32);
              *(_QWORD *)(v3 + 24) = 0;
              *(_QWORD *)(v3 + 16) = result;
              for ( i = result + 8 * v103; i != result; result += 8 )
              {
                if ( result )
                  *(_QWORD *)result = -8;
              }
              return result;
            }
LABEL_123:
            result = *(_QWORD *)(v3 + 16);
            for ( j = result + 8 * v94; j != result; result += 8 )
              *(_QWORD *)result = -8;
            *(_QWORD *)(v3 + 24) = 0;
            return result;
          }
LABEL_127:
          result = *(unsigned int *)(v3 + 28);
          if ( !(_DWORD)result )
            return result;
          v94 = *(unsigned int *)(v3 + 32);
          if ( (unsigned int)v94 > 0x40 )
          {
            result = j___libc_free_0(*(_QWORD *)(v3 + 16));
            *(_QWORD *)(v3 + 16) = 0;
            *(_QWORD *)(v3 + 24) = 0;
            *(_DWORD *)(v3 + 32) = 0;
            return result;
          }
          goto LABEL_123;
        }
        v55 = (4 * (int)v54 / 3u + 1) | ((unsigned __int64)(4 * (int)v54 / 3u + 1) >> 1);
        v56 = ((v55 | (v55 >> 2)) >> 4) | v55 | (v55 >> 2) | ((((v55 | (v55 >> 2)) >> 4) | v55 | (v55 >> 2)) >> 8);
        v57 = (v56 | (v56 >> 16)) + 1;
        v58 = 8 * ((v56 | (v56 >> 16)) + 1);
      }
      else
      {
        v58 = 1024;
        v57 = 128;
      }
      j___libc_free_0(v52);
      *(_DWORD *)(v3 + 32) = v57;
      result = sub_22077B0(v58);
      v59 = *(unsigned int *)(v3 + 32);
      *(_QWORD *)(v3 + 24) = 0;
      *(_QWORD *)(v3 + 16) = result;
      for ( k = result + 8 * v59; k != result; result += 8 )
      {
        if ( result )
          *(_QWORD *)result = -8;
      }
      ++*(_QWORD *)(v3 + 8);
      return result;
    }
LABEL_66:
    v61 = *(__int64 **)(v3 + 16);
    v62 = &v61[*(unsigned int *)(v3 + 32)];
    if ( v61 == v62 )
      goto LABEL_42;
    while ( *v61 == -8 || *v61 == -16 )
    {
      if ( v62 == ++v61 )
        goto LABEL_42;
    }
    if ( v62 == v61 )
      goto LABEL_42;
    v108 = v30 + 8;
    while ( 1 )
    {
      v63 = *v61;
      v64 = 0;
      v65 = *(unsigned int *)(*v61 + 64);
      v66 = 8 * v65;
      if ( (_DWORD)v65 )
      {
        do
        {
          while ( 1 )
          {
            v67 = (__int64 *)(v64 + *(_QWORD *)(v63 + 56));
            if ( v3 == *v67 )
              break;
            v64 += 8;
            if ( v64 == v66 )
              goto LABEL_78;
          }
          v64 += 8;
          *v67 = v30;
        }
        while ( v64 != v66 );
      }
LABEL_78:
      v68 = *(_DWORD *)(v30 + 32);
      if ( v68 )
      {
        v69 = *(_QWORD *)(v30 + 16);
        v70 = (v68 - 1) & (((unsigned int)v63 >> 4) ^ ((unsigned int)v63 >> 9));
        v71 = (_QWORD *)(v69 + 8LL * v70);
        v72 = *v71;
        if ( v63 == *v71 )
          goto LABEL_80;
        v73 = 1;
        v74 = 0;
        while ( v72 != -8 )
        {
          if ( v72 != -16 || v74 )
            v71 = v74;
          v70 = (v68 - 1) & (v73 + v70);
          v72 = *(_QWORD *)(v69 + 8LL * v70);
          if ( v63 == v72 )
            goto LABEL_80;
          ++v73;
          v74 = v71;
          v71 = (_QWORD *)(v69 + 8LL * v70);
        }
        v75 = *(_DWORD *)(v30 + 24);
        if ( !v74 )
          v74 = v71;
        ++*(_QWORD *)(v30 + 8);
        v76 = v75 + 1;
        if ( 4 * (v75 + 1) < 3 * v68 )
        {
          if ( v68 - *(_DWORD *)(v30 + 28) - v76 <= v68 >> 3 )
          {
            v106 = ((unsigned int)v63 >> 4) ^ ((unsigned int)v63 >> 9);
            sub_1C4A480(v108, v68);
            v84 = *(_DWORD *)(v30 + 32);
            if ( !v84 )
            {
LABEL_158:
              ++*(_DWORD *)(v30 + 24);
              BUG();
            }
            v85 = v84 - 1;
            v86 = *(_QWORD *)(v30 + 16);
            v87 = 0;
            v88 = v85 & v106;
            v74 = (_QWORD *)(v86 + 8LL * (v85 & v106));
            v76 = *(_DWORD *)(v30 + 24) + 1;
            v89 = 1;
            v90 = *v74;
            if ( v63 != *v74 )
            {
              while ( v90 != -8 )
              {
                if ( v90 == -16 && !v87 )
                  v87 = v74;
                v88 = v85 & (v88 + v89);
                v74 = (_QWORD *)(v86 + 8LL * v88);
                v90 = *v74;
                if ( v63 == *v74 )
                  goto LABEL_93;
                ++v89;
              }
              if ( v87 )
                v74 = v87;
            }
          }
          goto LABEL_93;
        }
      }
      else
      {
        ++*(_QWORD *)(v30 + 8);
      }
      sub_1C4A480(v108, 2 * v68);
      v77 = *(_DWORD *)(v30 + 32);
      if ( !v77 )
        goto LABEL_158;
      v78 = v77 - 1;
      v79 = *(_QWORD *)(v30 + 16);
      LODWORD(v80) = (v77 - 1) & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
      v76 = *(_DWORD *)(v30 + 24) + 1;
      v74 = (_QWORD *)(v79 + 8LL * (unsigned int)v80);
      v81 = *v74;
      if ( v63 != *v74 )
      {
        v82 = 1;
        v83 = 0;
        while ( v81 != -8 )
        {
          if ( !v83 && v81 == -16 )
            v83 = v74;
          v80 = v78 & (unsigned int)(v80 + v82);
          v74 = (_QWORD *)(v79 + 8 * v80);
          v81 = *v74;
          if ( v63 == *v74 )
            goto LABEL_93;
          ++v82;
        }
        if ( v83 )
          v74 = v83;
      }
LABEL_93:
      *(_DWORD *)(v30 + 24) = v76;
      if ( *v74 != -8 )
        --*(_DWORD *)(v30 + 28);
      *v74 = v63;
LABEL_80:
      if ( ++v61 != v62 )
      {
        while ( *v61 == -16 || *v61 == -8 )
        {
          if ( v62 == ++v61 )
            goto LABEL_42;
        }
        if ( v62 != v61 )
          continue;
      }
      goto LABEL_42;
    }
  }
  return result;
}
