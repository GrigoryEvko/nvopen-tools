// Function: sub_CE6510
// Address: 0xce6510
//
_BYTE *__fastcall sub_CE6510(__int64 *a1)
{
  int v1; // r14d
  __int64 *v2; // r12
  __int64 v3; // rsi
  _BYTE *v4; // r8
  __int64 v5; // r11
  _BYTE *v6; // r13
  __int64 v7; // r9
  unsigned int v8; // edi
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  unsigned int v14; // eax
  __int64 v15; // rbx
  int v16; // esi
  __int64 v17; // r9
  unsigned int v18; // ecx
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rdx
  _QWORD *v24; // rax
  _QWORD *i; // rdx
  __int64 v26; // rax
  int v27; // r15d
  _QWORD *v28; // rbx
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // r14
  _QWORD *v32; // r13
  __int64 v33; // r14
  __int64 v34; // rdi
  __int64 v35; // rdi
  _BYTE *result; // rax
  int v37; // eax
  int v38; // ecx
  int v39; // ecx
  __int64 v40; // rdi
  __int64 v41; // r9
  unsigned int v42; // r15d
  int v43; // r10d
  unsigned int v44; // ecx
  unsigned int v45; // eax
  _QWORD *v46; // rdi
  int v47; // ebx
  _QWORD *v48; // rax
  _QWORD *v49; // r12
  __int64 v50; // r13
  __int64 v51; // rdi
  __int64 v52; // rdi
  unsigned int v53; // edx
  int v54; // ebx
  unsigned int v55; // r15d
  unsigned int v56; // eax
  _QWORD *v57; // rdi
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rdi
  _QWORD *v60; // rax
  __int64 v61; // rdx
  _QWORD *k; // rdx
  unsigned __int64 v63; // rax
  unsigned __int64 v64; // rdi
  _QWORD *v65; // rax
  __int64 v66; // rdx
  _QWORD *j; // rdx
  int v68; // r15d
  __int64 v69; // r10
  _QWORD *v70; // rax
  _BYTE *v71; // [rsp+8h] [rbp-98h]
  _BYTE *v72; // [rsp+8h] [rbp-98h]
  __int64 v73; // [rsp+10h] [rbp-90h]
  int v74; // [rsp+10h] [rbp-90h]
  __int64 v75; // [rsp+10h] [rbp-90h]
  __int64 *v76; // [rsp+10h] [rbp-90h]
  _BYTE *v77; // [rsp+20h] [rbp-80h] BYREF
  __int64 v78; // [rsp+28h] [rbp-78h]
  _BYTE v79[112]; // [rsp+30h] [rbp-70h] BYREF

  v1 = 0;
  v2 = a1;
  v3 = *a1;
  v77 = v79;
  v78 = 0x800000000LL;
  sub_CE3B30((__int64)&v77, v3);
  v4 = v77;
  v5 = (__int64)(a1 + 18);
  v6 = &v77[8 * (unsigned int)v78];
  if ( v77 != v6 )
  {
    while ( 1 )
    {
      v11 = *((_QWORD *)v6 - 1);
      v12 = v2[2];
      if ( v11 )
      {
        v13 = (unsigned int)(*(_DWORD *)(v11 + 44) + 1);
        v14 = *(_DWORD *)(v11 + 44) + 1;
      }
      else
      {
        v13 = 0;
        v14 = 0;
      }
      v15 = 0;
      if ( v14 < *(_DWORD *)(v12 + 32) )
        v15 = *(_QWORD *)(*(_QWORD *)(v12 + 24) + 8 * v13);
      v3 = *((unsigned int *)v2 + 42);
      ++v1;
      if ( !(_DWORD)v3 )
        break;
      v7 = v2[19];
      v8 = (v3 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( v15 == *v9 )
      {
LABEL_4:
        v6 -= 8;
        *((_DWORD *)v9 + 2) = v1;
        if ( v4 == v6 )
          goto LABEL_16;
      }
      else
      {
        v74 = 1;
        v20 = 0;
        while ( v10 != -4096 )
        {
          if ( v10 == -8192 && !v20 )
            v20 = (__int64)v9;
          v8 = (v3 - 1) & (v74 + v8);
          v9 = (__int64 *)(v7 + 16LL * v8);
          v10 = *v9;
          if ( v15 == *v9 )
            goto LABEL_4;
          ++v74;
        }
        if ( !v20 )
          v20 = (__int64)v9;
        v37 = *((_DWORD *)v2 + 40);
        ++v2[18];
        v19 = v37 + 1;
        if ( 4 * v19 < (unsigned int)(3 * v3) )
        {
          if ( (int)v3 - *((_DWORD *)v2 + 41) - v19 <= (unsigned int)v3 >> 3 )
          {
            v72 = v4;
            v75 = v5;
            sub_CE3370(v5, v3);
            v38 = *((_DWORD *)v2 + 42);
            if ( !v38 )
            {
LABEL_123:
              ++*((_DWORD *)v2 + 40);
              BUG();
            }
            v39 = v38 - 1;
            v40 = v2[19];
            v41 = 0;
            v42 = v39 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v5 = v75;
            v4 = v72;
            v43 = 1;
            v19 = *((_DWORD *)v2 + 40) + 1;
            v20 = v40 + 16LL * v42;
            v3 = *(_QWORD *)v20;
            if ( v15 != *(_QWORD *)v20 )
            {
              while ( v3 != -4096 )
              {
                if ( v3 == -8192 && !v41 )
                  v41 = v20;
                v42 = v39 & (v43 + v42);
                v20 = v40 + 16LL * v42;
                v3 = *(_QWORD *)v20;
                if ( v15 == *(_QWORD *)v20 )
                  goto LABEL_13;
                ++v43;
              }
              if ( v41 )
                v20 = v41;
            }
          }
          goto LABEL_13;
        }
LABEL_11:
        v71 = v4;
        v73 = v5;
        sub_CE3370(v5, 2 * v3);
        v16 = *((_DWORD *)v2 + 42);
        if ( !v16 )
          goto LABEL_123;
        v3 = (unsigned int)(v16 - 1);
        v17 = v2[19];
        v5 = v73;
        v4 = v71;
        v18 = v3 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v19 = *((_DWORD *)v2 + 40) + 1;
        v20 = v17 + 16LL * v18;
        v21 = *(_QWORD *)v20;
        if ( v15 != *(_QWORD *)v20 )
        {
          v68 = 1;
          v69 = 0;
          while ( v21 != -4096 )
          {
            if ( v21 == -8192 && !v69 )
              v69 = v20;
            v18 = v3 & (v68 + v18);
            v20 = v17 + 16LL * v18;
            v21 = *(_QWORD *)v20;
            if ( v15 == *(_QWORD *)v20 )
              goto LABEL_13;
            ++v68;
          }
          if ( v69 )
            v20 = v69;
        }
LABEL_13:
        *((_DWORD *)v2 + 40) = v19;
        if ( *(_QWORD *)v20 != -4096 )
          --*((_DWORD *)v2 + 41);
        v6 -= 8;
        *(_QWORD *)v20 = v15;
        *(_DWORD *)(v20 + 8) = 0;
        *(_DWORD *)(v20 + 8) = v1;
        if ( v4 == v6 )
          goto LABEL_16;
      }
    }
    ++v2[18];
    goto LABEL_11;
  }
LABEL_16:
  sub_FCD9F0(v2);
  v22 = *((_DWORD *)v2 + 18);
  ++v2[7];
  if ( !v22 )
  {
    if ( !*((_DWORD *)v2 + 19) )
      goto LABEL_22;
    v23 = *((unsigned int *)v2 + 20);
    if ( (unsigned int)v23 > 0x40 )
    {
      v3 = 16LL * (unsigned int)v23;
      sub_C7D6A0(v2[8], v3, 8);
      v2[8] = 0;
      v2[9] = 0;
      *((_DWORD *)v2 + 20) = 0;
      goto LABEL_22;
    }
    goto LABEL_19;
  }
  v44 = 4 * v22;
  v3 = 64;
  v23 = *((unsigned int *)v2 + 20);
  if ( (unsigned int)(4 * v22) < 0x40 )
    v44 = 64;
  if ( v44 >= (unsigned int)v23 )
  {
LABEL_19:
    v24 = (_QWORD *)v2[8];
    for ( i = &v24[2 * v23]; i != v24; v24 += 2 )
      *v24 = -4096;
    v2[9] = 0;
    goto LABEL_22;
  }
  v45 = v22 - 1;
  if ( v45 )
  {
    _BitScanReverse(&v45, v45);
    v46 = (_QWORD *)v2[8];
    v47 = 1 << (33 - (v45 ^ 0x1F));
    if ( v47 < 64 )
      v47 = 64;
    if ( (_DWORD)v23 == v47 )
    {
      v2[9] = 0;
      v48 = &v46[2 * (unsigned int)v23];
      do
      {
        if ( v46 )
          *v46 = -4096;
        v46 += 2;
      }
      while ( v48 != v46 );
      goto LABEL_22;
    }
  }
  else
  {
    v46 = (_QWORD *)v2[8];
    v47 = 64;
  }
  sub_C7D6A0((__int64)v46, 16LL * (unsigned int)v23, 8);
  v3 = 8;
  v63 = ((((((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
           | (4 * v47 / 3u + 1)
           | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4)
         | (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
         | (4 * v47 / 3u + 1)
         | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
         | (4 * v47 / 3u + 1)
         | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4)
       | (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
       | (4 * v47 / 3u + 1)
       | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 16;
  v64 = (v63
       | (((((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
           | (4 * v47 / 3u + 1)
           | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4)
         | (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
         | (4 * v47 / 3u + 1)
         | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
         | (4 * v47 / 3u + 1)
         | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 4)
       | (((4 * v47 / 3u + 1) | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1)) >> 2)
       | (4 * v47 / 3u + 1)
       | ((unsigned __int64)(4 * v47 / 3u + 1) >> 1))
      + 1;
  *((_DWORD *)v2 + 20) = v64;
  v65 = (_QWORD *)sub_C7D670(16 * v64, 8);
  v66 = *((unsigned int *)v2 + 20);
  v2[9] = 0;
  v2[8] = (__int64)v65;
  for ( j = &v65[2 * v66]; j != v65; v65 += 2 )
  {
    if ( v65 )
      *v65 = -4096;
  }
LABEL_22:
  v26 = v2[11];
  if ( v26 != v2[12] )
    v2[12] = v26;
  v27 = *((_DWORD *)v2 + 32);
  ++v2[14];
  if ( v27 || *((_DWORD *)v2 + 33) )
  {
    v28 = (_QWORD *)v2[15];
    v29 = 4 * v27;
    v30 = *((unsigned int *)v2 + 34);
    v31 = 16 * v30;
    if ( (unsigned int)(4 * v27) < 0x40 )
      v29 = 64;
    v32 = &v28[(unsigned __int64)v31 / 8];
    if ( v29 >= (unsigned int)v30 )
    {
      while ( v28 != v32 )
      {
        if ( *v28 != -4096 )
        {
          if ( *v28 != -8192 )
          {
            v33 = v28[1];
            if ( v33 )
            {
              v34 = *(_QWORD *)(v33 + 96);
              if ( v34 != v33 + 112 )
                _libc_free(v34, v3);
              v35 = *(_QWORD *)(v33 + 24);
              if ( v35 != v33 + 40 )
                _libc_free(v35, v3);
              v3 = 168;
              j_j___libc_free_0(v33, 168);
            }
          }
          *v28 = -4096;
        }
        v28 += 2;
      }
    }
    else
    {
      v76 = v2;
      v49 = &v28[(unsigned __int64)v31 / 8];
      do
      {
        if ( *v28 != -8192 && *v28 != -4096 )
        {
          v50 = v28[1];
          if ( v50 )
          {
            v51 = *(_QWORD *)(v50 + 96);
            if ( v51 != v50 + 112 )
              _libc_free(v51, v3);
            v52 = *(_QWORD *)(v50 + 24);
            if ( v52 != v50 + 40 )
              _libc_free(v52, v3);
            v3 = 168;
            j_j___libc_free_0(v50, 168);
          }
        }
        v28 += 2;
      }
      while ( v28 != v49 );
      v2 = v76;
      v53 = *((_DWORD *)v76 + 34);
      if ( v27 )
      {
        v54 = 64;
        v55 = v27 - 1;
        if ( v55 )
        {
          _BitScanReverse(&v56, v55);
          v54 = 1 << (33 - (v56 ^ 0x1F));
          if ( v54 < 64 )
            v54 = 64;
        }
        v57 = (_QWORD *)v76[15];
        if ( v53 == v54 )
        {
          v76[16] = 0;
          v70 = &v57[2 * v53];
          do
          {
            if ( v57 )
              *v57 = -4096;
            v57 += 2;
          }
          while ( v70 != v57 );
        }
        else
        {
          sub_C7D6A0((__int64)v57, v31, 8);
          v3 = 8;
          v58 = ((((((((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
                   | (4 * v54 / 3u + 1)
                   | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
                 | (4 * v54 / 3u + 1)
                 | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
                 | (4 * v54 / 3u + 1)
                 | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 4)
               | (((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
               | (4 * v54 / 3u + 1)
               | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 16;
          v59 = (v58
               | (((((((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
                   | (4 * v54 / 3u + 1)
                   | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
                 | (4 * v54 / 3u + 1)
                 | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
                 | (4 * v54 / 3u + 1)
                 | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 4)
               | (((4 * v54 / 3u + 1) | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1)) >> 2)
               | (4 * v54 / 3u + 1)
               | ((unsigned __int64)(4 * v54 / 3u + 1) >> 1))
              + 1;
          *((_DWORD *)v76 + 34) = v59;
          v60 = (_QWORD *)sub_C7D670(16 * v59, 8);
          v61 = *((unsigned int *)v76 + 34);
          v76[16] = 0;
          v76[15] = (__int64)v60;
          for ( k = &v60[2 * v61]; k != v60; v60 += 2 )
          {
            if ( v60 )
              *v60 = -4096;
          }
        }
        goto LABEL_42;
      }
      if ( v53 )
      {
        v3 = v31;
        sub_C7D6A0(v76[15], v31, 8);
        v76[15] = 0;
        v76[16] = 0;
        *((_DWORD *)v76 + 34) = 0;
        goto LABEL_42;
      }
    }
    v2[16] = 0;
  }
LABEL_42:
  v2[3] = 0;
  v2[4] = 0;
  sub_FCF420(v2);
  result = sub_CE6110(v2);
  if ( v77 != v79 )
    return (_BYTE *)_libc_free(v77, v3);
  return result;
}
