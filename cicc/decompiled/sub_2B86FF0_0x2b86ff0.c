// Function: sub_2B86FF0
// Address: 0x2b86ff0
//
void __fastcall sub_2B86FF0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned int v8; // edx
  __int64 v9; // r12
  __int64 v10; // rcx
  __int64 v11; // r14
  __int64 v12; // rsi
  __int64 v13; // r8
  unsigned int v14; // edx
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 i; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r13
  __int64 v21; // r11
  int v22; // r15d
  __int64 v23; // rdx
  int v24; // r15d
  __int64 v25; // rdx
  __int64 v26; // r11
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r12
  _BYTE *v32; // r14
  __int64 *v33; // r15
  unsigned __int64 v34; // rdx
  unsigned int *v35; // rcx
  unsigned int *v36; // r13
  unsigned int *v37; // rbx
  unsigned __int64 v38; // r12
  __int64 v39; // rdi
  _BYTE *v40; // rbx
  unsigned __int64 v41; // rdi
  __int64 v42; // r14
  __int64 v43; // rax
  __int64 v44; // r9
  unsigned __int64 v45; // r15
  __int64 v46; // r13
  int v47; // r8d
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rdx
  __int64 *v50; // rax
  __int64 v51; // rdx
  unsigned __int64 v52; // r15
  __int64 *v53; // rax
  __int64 *j; // rdx
  __int64 *v55; // rdx
  __int64 v56; // r12
  __int64 v57; // r14
  __int64 v58; // rdi
  __int64 v59; // rdx
  __int64 v60; // r13
  __int64 v61; // rax
  __int64 v62; // rsi
  __int64 v63; // rdx
  __int64 v64; // r15
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  __int64 v67; // r13
  unsigned __int64 v68; // rax
  unsigned __int64 v69; // rcx
  __int64 *v70; // rax
  __int64 v71; // rcx
  unsigned __int64 v72; // rdx
  __int64 *v73; // rax
  __int64 *k; // rdx
  int v75; // r14d
  __int64 v76; // rdx
  char v77; // r10
  __int64 *v78; // r13
  int v79; // r11d
  __int64 *v80; // rcx
  __int64 v81; // r15
  __int64 v82; // rsi
  _BYTE *v83; // rcx
  __int64 v84; // rsi
  _QWORD *v85; // rsi
  __int64 v86; // r11
  __int64 v87; // r11
  int v88; // ecx
  unsigned __int64 v89; // rdx
  __int64 *v90; // rax
  __int64 v91; // rdx
  __int64 *v92; // rax
  __int64 v93; // rdx
  int v94; // [rsp+Ch] [rbp-1E4h]
  __int64 v95; // [rsp+18h] [rbp-1D8h]
  unsigned int v96; // [rsp+18h] [rbp-1D8h]
  __int64 v97; // [rsp+20h] [rbp-1D0h]
  unsigned __int64 v98; // [rsp+20h] [rbp-1D0h]
  __int64 v99; // [rsp+28h] [rbp-1C8h]
  _BYTE *v100; // [rsp+28h] [rbp-1C8h]
  __int64 v101; // [rsp+38h] [rbp-1B8h] BYREF
  __int64 v102; // [rsp+40h] [rbp-1B0h] BYREF
  __int64 v103; // [rsp+48h] [rbp-1A8h]
  __int64 *v104; // [rsp+50h] [rbp-1A0h] BYREF
  unsigned int v105; // [rsp+58h] [rbp-198h]
  _BYTE *v106; // [rsp+90h] [rbp-160h] BYREF
  __int64 v107; // [rsp+98h] [rbp-158h]
  _BYTE v108[336]; // [rsp+A0h] [rbp-150h] BYREF

  v7 = a1[1];
  v8 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
  if ( v8 <= 4 )
  {
    if ( !v8 )
      return;
    v9 = 0;
    v99 = v8;
    while ( 1 )
    {
      v10 = *a1;
      v11 = 8LL * (unsigned int)v9;
      v12 = *(_QWORD *)(*(_QWORD *)(v7 - 8) + 32LL * *(unsigned int *)(v7 + 72) + v11);
      if ( v12 )
      {
        v13 = (unsigned int)(*(_DWORD *)(v12 + 44) + 1);
        v14 = *(_DWORD *)(v12 + 44) + 1;
      }
      else
      {
        v13 = 0;
        v14 = 0;
      }
      if ( v14 < *(_DWORD *)(v10 + 32) && *(_QWORD *)(*(_QWORD *)(v10 + 24) + 8 * v13) )
      {
        v15 = a1[2];
        v16 = 8LL * *((unsigned int *)a1 + 6);
        if ( v16 )
        {
          for ( i = 0; i != v16; i += 8 )
          {
            while ( 1 )
            {
              v18 = *(_QWORD *)(v15 + i);
              v19 = a1[10];
              if ( *(_BYTE *)v18 == 84 )
                break;
              *(_QWORD *)(*(_QWORD *)(v19 + ((unsigned __int64)(unsigned int)v9 << 6)) + i) = v18;
LABEL_11:
              i += 8;
              if ( i == v16 )
                goto LABEL_20;
            }
            v20 = *(_QWORD *)(v18 - 8);
            v21 = 32LL * *(unsigned int *)(v18 + 72);
            if ( v12 == *(_QWORD *)(v20 + v11 + v21) )
            {
              *(_QWORD *)(*(_QWORD *)(v19 + ((unsigned __int64)(unsigned int)v9 << 6)) + i) = *(_QWORD *)(v20 + 32LL * (unsigned int)v9);
              goto LABEL_11;
            }
            v22 = *(_DWORD *)(v18 + 4);
            v23 = 0x1FFFFFFFE0LL;
            v24 = v22 & 0x7FFFFFF;
            if ( v24 )
            {
              v25 = 0;
              v26 = v20 + v21;
              do
              {
                if ( v12 == *(_QWORD *)(v26 + 8 * v25) )
                {
                  v23 = 32 * v25;
                  goto LABEL_19;
                }
                ++v25;
              }
              while ( v24 != (_DWORD)v25 );
              v23 = 0x1FFFFFFFE0LL;
            }
LABEL_19:
            *(_QWORD *)(*(_QWORD *)(v19 + ((unsigned __int64)(unsigned int)v9 << 6)) + i) = *(_QWORD *)(v20 + v23);
          }
        }
      }
      else
      {
        v42 = a1[10] + ((unsigned __int64)(unsigned int)v9 << 6);
        v43 = sub_ACADE0(*(__int64 ***)(v7 + 8));
        v45 = *((unsigned int *)a1 + 6);
        v46 = v43;
        v47 = v45;
        if ( *(_DWORD *)(v42 + 12) < (unsigned int)v45 )
        {
          v89 = *((unsigned int *)a1 + 6);
          *(_DWORD *)(v42 + 8) = 0;
          sub_C8D5F0(v42, (const void *)(v42 + 16), v89, 8u, v45, v44);
          v90 = *(__int64 **)v42;
          v91 = *(_QWORD *)v42 + 8 * v45;
          do
            *v90++ = v46;
          while ( (__int64 *)v91 != v90 );
          *(_DWORD *)(v42 + 8) = v45;
        }
        else
        {
          v48 = *(unsigned int *)(v42 + 8);
          v49 = v48;
          if ( v45 <= v48 )
            v49 = *((unsigned int *)a1 + 6);
          if ( v49 )
          {
            v50 = *(__int64 **)v42;
            v51 = *(_QWORD *)v42 + 8 * v49;
            do
              *v50++ = v46;
            while ( (__int64 *)v51 != v50 );
            v48 = *(unsigned int *)(v42 + 8);
          }
          if ( v45 > v48 )
          {
            v52 = v45 - v48;
            if ( v52 )
            {
              v53 = (__int64 *)(*(_QWORD *)v42 + 8 * v48);
              for ( j = &v53[v52]; j != v53; ++v53 )
                *v53 = v46;
            }
          }
          *(_DWORD *)(v42 + 8) = v47;
        }
      }
LABEL_20:
      if ( ++v9 == v99 )
        return;
      v7 = a1[1];
    }
  }
  v102 = 0;
  v103 = 1;
  v55 = (__int64 *)&v104;
  do
  {
    *v55 = -4096;
    v55 += 2;
  }
  while ( v55 != (__int64 *)&v106 );
  v106 = v108;
  v107 = 0x400000000LL;
  v56 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
  if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) == 0 )
  {
    v97 = a1[2];
    v95 = 8LL * *((unsigned int *)a1 + 6);
    if ( !v95 )
    {
      v32 = v108;
      goto LABEL_37;
    }
LABEL_83:
    v31 = 0;
    while ( 1 )
    {
      v29 = *(_QWORD *)(v97 + v31);
      if ( *(_BYTE *)v29 == 13 )
      {
        v27 = 0;
        v28 = *(_DWORD *)(a1[1] + 4) & 0x7FFFFFF;
        if ( (*(_DWORD *)(a1[1] + 4) & 0x7FFFFFF) != 0 )
        {
          while ( 1 )
          {
            v30 = (unsigned int)v27++;
            *(_QWORD *)(*(_QWORD *)(a1[10] + (v30 << 6)) + v31) = v29;
            if ( v28 == v27 )
              break;
            v29 = *(_QWORD *)(v97 + v31);
          }
        }
      }
      else
      {
        v75 = *(_DWORD *)(v29 + 4) & 0x7FFFFFF;
        if ( v75 )
        {
          v76 = 0;
          while ( 1 )
          {
            a6 = *(_QWORD *)(v29 - 8);
            a5 = 8LL * (unsigned int)v76;
            v84 = *(_QWORD *)(a6 + 32LL * *(unsigned int *)(v29 + 72) + a5);
            if ( *(_QWORD *)(*(_QWORD *)(a1[1] - 8) + 32LL * *(unsigned int *)(a1[1] + 72) + a5) != v84 )
              break;
            v85 = (_QWORD *)(v31 + *(_QWORD *)(a1[10] + ((unsigned __int64)(unsigned int)v76 << 6)));
            if ( !*v85 || *(_BYTE *)*v85 != 13 )
              *v85 = *(_QWORD *)(a6 + 32LL * (unsigned int)v76);
LABEL_95:
            if ( v75 == ++v76 )
              goto LABEL_26;
          }
          v77 = v103 & 1;
          if ( (v103 & 1) != 0 )
          {
            v78 = (__int64 *)&v104;
            v79 = 3;
            goto LABEL_89;
          }
          v86 = v105;
          v78 = v104;
          if ( v105 )
          {
            v79 = v105 - 1;
LABEL_89:
            a5 = v79 & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
            v80 = &v78[2 * a5];
            v81 = *v80;
            if ( v84 == *v80 )
            {
LABEL_90:
              v82 = 8;
              if ( !v77 )
                v82 = 2LL * v105;
              if ( v80 != &v78[v82] )
              {
                a5 = 9LL * (unsigned int)v107;
                v83 = &v106[72 * *((unsigned int *)v80 + 2)];
                if ( v83 != &v106[72 * (unsigned int)v107] )
                  *(_QWORD *)(*(_QWORD *)(a1[10] + ((unsigned __int64)**((unsigned int **)v83 + 1) << 6)) + v31) = *(_QWORD *)(a6 + 32LL * (unsigned int)v76);
              }
              goto LABEL_95;
            }
            v88 = 1;
            while ( v81 != -4096 )
            {
              a5 = v79 & (unsigned int)(v88 + a5);
              v94 = v88 + 1;
              v80 = &v78[2 * (unsigned int)a5];
              v81 = *v80;
              if ( v84 == *v80 )
                goto LABEL_90;
              v88 = v94;
            }
            if ( v77 )
            {
              v87 = 8;
              goto LABEL_104;
            }
            v86 = v105;
          }
          v87 = 2 * v86;
LABEL_104:
          v80 = &v78[v87];
          goto LABEL_90;
        }
      }
LABEL_26:
      v31 += 8;
      if ( v95 == v31 )
        goto LABEL_27;
    }
  }
  v57 = 0;
  while ( 1 )
  {
    v62 = *a1;
    v63 = *(_QWORD *)(*(_QWORD *)(v7 - 8) + 32LL * *(unsigned int *)(v7 + 72) + 8LL * (unsigned int)v57);
    v101 = v63;
    if ( v63 )
      break;
    v58 = 0;
    if ( *(_DWORD *)(v62 + 32) )
      goto LABEL_62;
LABEL_69:
    v64 = a1[10] + ((unsigned __int64)(unsigned int)v57 << 6);
    v65 = sub_ACADE0(*(__int64 ***)(v7 + 8));
    v66 = *((unsigned int *)a1 + 6);
    v67 = v65;
    a5 = v66;
    if ( *(_DWORD *)(v64 + 12) < (unsigned int)v66 )
    {
      *(_DWORD *)(v64 + 8) = 0;
      v96 = v66;
      v98 = v66;
      sub_C8D5F0(v64, (const void *)(v64 + 16), v66, 8u, v66, a6);
      v92 = *(__int64 **)v64;
      a5 = v96;
      v93 = *(_QWORD *)v64 + 8 * v98;
      do
        *v92++ = v67;
      while ( (__int64 *)v93 != v92 );
      *(_DWORD *)(v64 + 8) = v96;
    }
    else
    {
      v68 = *(unsigned int *)(v64 + 8);
      v69 = v68;
      if ( v66 <= v68 )
        v69 = *((unsigned int *)a1 + 6);
      if ( v69 )
      {
        v70 = *(__int64 **)v64;
        v71 = *(_QWORD *)v64 + 8 * v69;
        do
          *v70++ = v67;
        while ( (__int64 *)v71 != v70 );
        v68 = *(unsigned int *)(v64 + 8);
      }
      if ( v66 > v68 )
      {
        v72 = v66 - v68;
        if ( v72 )
        {
          v73 = (__int64 *)(*(_QWORD *)v64 + 8 * v68);
          for ( k = &v73[v72]; k != v73; ++v73 )
            *v73 = v67;
        }
      }
      *(_DWORD *)(v64 + 8) = a5;
    }
    if ( v56 == ++v57 )
      goto LABEL_82;
LABEL_66:
    v7 = a1[1];
  }
  v58 = (unsigned int)(*(_DWORD *)(v63 + 44) + 1);
  if ( (unsigned int)(*(_DWORD *)(v63 + 44) + 1) >= *(_DWORD *)(v62 + 32) )
    goto LABEL_69;
LABEL_62:
  v59 = *(_QWORD *)(v62 + 24);
  if ( !*(_QWORD *)(v59 + 8 * v58) )
    goto LABEL_69;
  v60 = sub_2B86B60((__int64)&v102, &v101, v59, (unsigned int)v57, a5, a6);
  v61 = *(unsigned int *)(v60 + 16);
  if ( v61 + 1 > (unsigned __int64)*(unsigned int *)(v60 + 20) )
  {
    sub_C8D5F0(v60 + 8, (const void *)(v60 + 24), v61 + 1, 4u, a5, a6);
    v61 = *(unsigned int *)(v60 + 16);
  }
  *(_DWORD *)(*(_QWORD *)(v60 + 8) + 4 * v61) = v57++;
  ++*(_DWORD *)(v60 + 16);
  if ( v56 != v57 )
    goto LABEL_66;
LABEL_82:
  v97 = a1[2];
  v95 = 8LL * *((unsigned int *)a1 + 6);
  if ( v95 )
    goto LABEL_83;
LABEL_27:
  v32 = v106;
  v33 = a1;
  v100 = &v106[72 * (unsigned int)v107];
  if ( v100 != v106 )
  {
    do
    {
      v34 = *((unsigned int *)v32 + 4);
      if ( v34 > 1 )
      {
        v35 = (unsigned int *)*((_QWORD *)v32 + 1);
        v36 = &v35[v34];
        v37 = v35 + 1;
        v38 = (unsigned __int64)*v35 << 6;
        do
        {
          v39 = *v37++;
          sub_2B0CFB0(v33[10] + (v39 << 6), v38 + v33[10], v34, (__int64)v35, a5, a6);
        }
        while ( v36 != v37 );
      }
      v32 += 72;
    }
    while ( v32 != v100 );
    v40 = v106;
    v32 = &v106[72 * (unsigned int)v107];
    if ( v106 != v32 )
    {
      do
      {
        v32 -= 72;
        v41 = *((_QWORD *)v32 + 1);
        if ( (_BYTE *)v41 != v32 + 24 )
          _libc_free(v41);
      }
      while ( v40 != v32 );
      v32 = v106;
    }
  }
LABEL_37:
  if ( v32 != v108 )
    _libc_free((unsigned __int64)v32);
  if ( (v103 & 1) == 0 )
    sub_C7D6A0((__int64)v104, 16LL * v105, 8);
}
