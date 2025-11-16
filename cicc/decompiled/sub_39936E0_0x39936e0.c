// Function: sub_39936E0
// Address: 0x39936e0
//
void __fastcall sub_39936E0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *v4; // rax
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v9; // r14
  unsigned int v10; // r8d
  __int64 v11; // r11
  int v12; // r9d
  unsigned int v13; // ecx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 *v16; // rdx
  unsigned __int64 v17; // rbx
  unsigned int j; // edi
  __int64 *v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // rdi
  _QWORD *v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // r9
  _QWORD *v25; // r10
  __int64 v26; // rbx
  _QWORD *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v30; // rax
  _QWORD *v31; // rdx
  unsigned __int64 *v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // r8
  unsigned int v35; // edi
  unsigned __int64 v36; // rsi
  unsigned __int64 v37; // rsi
  unsigned int m; // eax
  unsigned __int64 *v39; // rsi
  int v40; // eax
  __int64 v41; // rdi
  unsigned int v42; // eax
  __int64 *v43; // rdx
  int v44; // ecx
  int v45; // ecx
  __int64 v46; // rdi
  int v47; // r10d
  unsigned int v48; // esi
  __int64 *v49; // r8
  unsigned __int64 v50; // rdx
  unsigned __int64 v51; // rdx
  unsigned int i; // eax
  __int64 v53; // rsi
  unsigned int v54; // eax
  unsigned int v55; // edi
  int v56; // eax
  int v57; // eax
  int v58; // eax
  __int64 v59; // rsi
  int v60; // r8d
  __int64 *v61; // rdi
  unsigned int k; // ebx
  __int64 v63; // rcx
  unsigned int v64; // ebx
  __int64 v65; // [rsp+8h] [rbp-128h]
  __int64 v66; // [rsp+10h] [rbp-120h]
  __int64 v68; // [rsp+28h] [rbp-108h]
  _QWORD *v69; // [rsp+30h] [rbp-100h]
  __int64 v71[2]; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v72; // [rsp+50h] [rbp-E0h] BYREF
  char v73[48]; // [rsp+60h] [rbp-D0h] BYREF
  __m128i v74; // [rsp+90h] [rbp-A0h] BYREF
  unsigned __int64 *v75; // [rsp+A0h] [rbp-90h] BYREF
  int v76; // [rsp+A8h] [rbp-88h]
  char v77; // [rsp+100h] [rbp-30h] BYREF

  v4 = (unsigned __int64 *)&v75;
  v74.m128i_i64[0] = 0;
  v74.m128i_i64[1] = 1;
  do
  {
    *v4 = -8;
    v4 += 3;
    *(v4 - 2) = -8;
  }
  while ( v4 != (unsigned __int64 *)&v77 );
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 264LL);
  v6 = *(_QWORD *)(v5 + 608);
  v68 = v6 + 32LL * *(unsigned int *)(v5 + 616);
  while ( v68 != v6 )
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)v6;
      if ( *(_QWORD *)v6 )
      {
        v8 = *(_QWORD *)(v6 + 24);
        v9 = 0;
        if ( *(_DWORD *)(v8 + 8) == 2 )
          v9 = *(_QWORD *)(v8 - 8);
        v10 = *(_DWORD *)(a3 + 24);
        if ( !v10 )
        {
          ++*(_QWORD *)a3;
LABEL_49:
          sub_39929B0(a3, 2 * v10);
          v44 = *(_DWORD *)(a3 + 24);
          if ( !v44 )
            goto LABEL_89;
          v45 = v44 - 1;
          v47 = 1;
          v48 = (unsigned int)v9 >> 9;
          v49 = 0;
          v50 = (((v48 ^ ((unsigned int)v9 >> 4)
                 | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
                - 1
                - ((unsigned __int64)(v48 ^ ((unsigned int)v9 >> 4)) << 32)) >> 22)
              ^ ((v48 ^ ((unsigned int)v9 >> 4)
                | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
               - 1
               - ((unsigned __int64)(v48 ^ ((unsigned int)v9 >> 4)) << 32));
          v51 = ((9 * (((v50 - 1 - (v50 << 13)) >> 8) ^ (v50 - 1 - (v50 << 13)))) >> 15)
              ^ (9 * (((v50 - 1 - (v50 << 13)) >> 8) ^ (v50 - 1 - (v50 << 13))));
          for ( i = v45 & (((v51 - 1 - (v51 << 27)) >> 31) ^ (v51 - 1 - ((_DWORD)v51 << 27))); ; i = v45 & v54 )
          {
            v46 = *(_QWORD *)(a3 + 8);
            v16 = (__int64 *)(v46 + 16LL * i);
            v53 = *v16;
            if ( v7 == *v16 && v9 == v16[1] )
              break;
            if ( v53 == -8 )
            {
              if ( v16[1] == -8 )
              {
                v56 = *(_DWORD *)(a3 + 16) + 1;
                if ( v49 )
                  v16 = v49;
                goto LABEL_63;
              }
            }
            else if ( v53 == -16 && v16[1] == -16 && !v49 )
            {
              v49 = (__int64 *)(v46 + 16LL * i);
            }
            v54 = v47 + i;
            ++v47;
          }
LABEL_69:
          v56 = *(_DWORD *)(a3 + 16) + 1;
          goto LABEL_63;
        }
        v11 = *(_QWORD *)(a3 + 8);
        v12 = 1;
        v13 = (unsigned int)v9 >> 9;
        v14 = (((v13 ^ ((unsigned int)v9 >> 4)
               | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
              - 1
              - ((unsigned __int64)(v13 ^ ((unsigned int)v9 >> 4)) << 32)) >> 22)
            ^ ((v13 ^ ((unsigned int)v9 >> 4)
              | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
             - 1
             - ((unsigned __int64)(v13 ^ ((unsigned int)v9 >> 4)) << 32));
        v15 = ((v14 - 1 - (v14 << 13)) >> 8) ^ (v14 - 1 - (v14 << 13));
        v16 = 0;
        v17 = (((((9 * v15) >> 15) ^ (9 * v15)) - 1 - ((((9 * v15) >> 15) ^ (9 * v15)) << 27)) >> 31)
            ^ ((((9 * v15) >> 15) ^ (9 * v15)) - 1 - ((((9 * v15) >> 15) ^ (9 * v15)) << 27));
        for ( j = v17 & (v10 - 1); ; j = (v10 - 1) & v55 )
        {
          v19 = (__int64 *)(v11 + 16LL * j);
          v20 = *v19;
          if ( v7 == *v19 && v9 == v19[1] )
            goto LABEL_25;
          if ( v20 == -8 )
            break;
          if ( v20 == -16 && v19[1] == -16 && !v16 )
            v16 = (__int64 *)(v11 + 16LL * j);
LABEL_59:
          v55 = v12 + j;
          ++v12;
        }
        if ( v19[1] != -8 )
          goto LABEL_59;
        if ( !v16 )
          v16 = (__int64 *)(v11 + 16LL * j);
        ++*(_QWORD *)a3;
        v56 = *(_DWORD *)(a3 + 16) + 1;
        if ( 4 * v56 >= 3 * v10 )
          goto LABEL_49;
        if ( v10 - *(_DWORD *)(a3 + 20) - v56 <= v10 >> 3 )
        {
          sub_39929B0(a3, v10);
          v57 = *(_DWORD *)(a3 + 24);
          if ( v57 )
          {
            v58 = v57 - 1;
            v59 = *(_QWORD *)(a3 + 8);
            v60 = 1;
            v61 = 0;
            for ( k = v58 & v17; ; k = v58 & v64 )
            {
              v16 = (__int64 *)(v59 + 16LL * k);
              v63 = *v16;
              if ( v7 == *v16 && v9 == v16[1] )
                break;
              if ( v63 == -8 )
              {
                if ( v16[1] == -8 )
                {
                  v56 = *(_DWORD *)(a3 + 16) + 1;
                  if ( v61 )
                    v16 = v61;
                  goto LABEL_63;
                }
              }
              else if ( v63 == -16 && v16[1] == -16 && !v61 )
              {
                v61 = (__int64 *)(v59 + 16LL * k);
              }
              v64 = v60 + k;
              ++v60;
            }
            goto LABEL_69;
          }
LABEL_89:
          ++*(_DWORD *)(a3 + 16);
          BUG();
        }
LABEL_63:
        *(_DWORD *)(a3 + 16) = v56;
        if ( *v16 != -8 || v16[1] != -8 )
          --*(_DWORD *)(a3 + 20);
        *v16 = v7;
        v16[1] = v9;
        v8 = *(_QWORD *)(v6 + 24);
LABEL_25:
        v22 = sub_20FAEB0((_QWORD *)(a1 + 64), v8);
        if ( v22 )
          break;
      }
LABEL_20:
      v6 += 32;
      if ( v68 == v6 )
        goto LABEL_21;
    }
    v69 = v22;
    sub_398DFB0(a1, a2, v7, v9, v22[1]);
    v23 = (_QWORD *)sub_22077B0(0x48u);
    v25 = v69;
    v26 = (__int64)v23;
    if ( v23 )
    {
      *v23 = v7;
      v27 = v23 + 7;
      v28 = 0;
      *(v27 - 6) = v9;
      *(v27 - 5) = 0;
      *((_DWORD *)v27 - 8) = -1;
      *(v27 - 3) = 0;
      *(_QWORD *)(v26 + 40) = v27;
      *(_QWORD *)(v26 + 48) = 0x100000000LL;
      v29 = *(unsigned int *)(v6 + 16);
      v30 = *(_QWORD *)(v6 + 8);
    }
    else
    {
      v28 = MEMORY[0x30];
      v29 = *(unsigned int *)(v6 + 16);
      v30 = *(_QWORD *)(v6 + 8);
      if ( MEMORY[0x30] >= MEMORY[0x34] )
      {
        v65 = *(_QWORD *)(v6 + 8);
        v66 = *(unsigned int *)(v6 + 16);
        sub_16CD150(40, (const void *)0x38, 0, 16, v29, v24);
        v28 = MEMORY[0x30];
        v30 = v65;
        v29 = v66;
        v25 = v69;
      }
    }
    v31 = (_QWORD *)(*(_QWORD *)(v26 + 40) + 16 * v28);
    *v31 = v29;
    v31[1] = v30;
    ++*(_DWORD *)(v26 + 48);
    if ( (v74.m128i_i8[8] & 1) != 0 )
    {
      v32 = (unsigned __int64 *)&v75;
      v33 = 3;
LABEL_30:
      v34 = 1;
      v35 = (unsigned int)v9 >> 9;
      v36 = (((v35 ^ ((unsigned int)v9 >> 4)
             | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(v35 ^ ((unsigned int)v9 >> 4)) << 32)) >> 22)
          ^ ((v35 ^ ((unsigned int)v9 >> 4)
            | ((unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(v35 ^ ((unsigned int)v9 >> 4)) << 32));
      v37 = ((9 * (((v36 - 1 - (v36 << 13)) >> 8) ^ (v36 - 1 - (v36 << 13)))) >> 15)
          ^ (9 * (((v36 - 1 - (v36 << 13)) >> 8) ^ (v36 - 1 - (v36 << 13))));
      for ( m = v33 & (((v37 - 1 - (v37 << 27)) >> 31) ^ (v37 - 1 - ((_DWORD)v37 << 27))); ; m = v33 & v40 )
      {
        v39 = &v32[3 * m];
        if ( v7 == *v39 && v9 == v39[1] )
          break;
        if ( *v39 == -8 && v39[1] == -8 )
          goto LABEL_16;
        v40 = v34 + m;
        v34 = (unsigned int)(v34 + 1);
      }
      v41 = v39[2];
      if ( v41 )
      {
        sub_3988A40(v41, v26, v33, (__int64)v32, v34, v24);
        goto LABEL_17;
      }
    }
    else
    {
      v32 = v75;
      if ( v76 )
      {
        v33 = (unsigned int)(v76 - 1);
        goto LABEL_30;
      }
    }
LABEL_16:
    if ( !(unsigned __int8)sub_39A0D10(a1 + 4040, v25, v26) )
      goto LABEL_17;
    v71[0] = v7;
    v71[1] = v9;
    v72 = v26;
    sub_39931F0((__int64)v73, &v74, v71, &v72);
    v42 = *(_DWORD *)(a1 + 672);
    if ( v42 >= *(_DWORD *)(a1 + 676) )
    {
      sub_398E850(a1 + 664, 0);
      v42 = *(_DWORD *)(a1 + 672);
    }
    v43 = (__int64 *)(*(_QWORD *)(a1 + 664) + 8LL * v42);
    if ( !v43 )
    {
      *(_DWORD *)(a1 + 672) = v42 + 1;
LABEL_17:
      v21 = *(_QWORD *)(v26 + 40);
      if ( v21 != v26 + 56 )
        _libc_free(v21);
      j_j___libc_free_0(v26);
      goto LABEL_20;
    }
    *v43 = v26;
    v6 += 32;
    ++*(_DWORD *)(a1 + 672);
  }
LABEL_21:
  if ( (v74.m128i_i8[8] & 1) == 0 )
    j___libc_free_0((unsigned __int64)v75);
}
