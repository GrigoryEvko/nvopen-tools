// Function: sub_3588E20
// Address: 0x3588e20
//
void __fastcall sub_3588E20(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // r9
  int v5; // r10d
  __int64 v6; // r8
  __int64 *v7; // rdx
  __int64 v8; // rdi
  __int64 *v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // esi
  int v12; // r8d
  int v13; // ecx
  unsigned int v14; // eax
  __int64 v15; // r11
  int v16; // edi
  __int64 *v17; // rsi
  __int64 v18; // r15
  __int64 v19; // rcx
  int v20; // r10d
  __int64 *v21; // r8
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r14
  unsigned int v28; // esi
  int v29; // eax
  __int64 v30; // rcx
  int v31; // r10d
  __int64 v32; // r11
  int v33; // edx
  unsigned int v34; // eax
  __int64 v35; // r9
  int v36; // eax
  int v37; // eax
  int v38; // edi
  __int64 v39; // r10
  __int64 *v40; // r11
  int v41; // esi
  unsigned int v42; // eax
  int v43; // eax
  __int64 v44; // rdx
  __int64 v45; // rcx
  unsigned int v46; // eax
  unsigned int v47; // edx
  _QWORD *v48; // rdi
  __int64 i; // rax
  __int64 v50; // r13
  __int64 v51; // r15
  __int64 v52; // r15
  __int64 v53; // rbx
  __int64 v54; // r13
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  _QWORD *v57; // rcx
  unsigned __int64 j; // rax
  __int64 v59; // rcx
  int v60; // edi
  int v61; // edi
  int v62; // esi
  unsigned int v63; // r15d
  __int64 *v64; // rax
  int v65; // edi
  __int64 *v66; // rsi
  __int64 v68; // [rsp+8h] [rbp-108h]
  __int64 v69; // [rsp+20h] [rbp-F0h]
  __int64 v70; // [rsp+28h] [rbp-E8h]
  __int64 v71; // [rsp+38h] [rbp-D8h] BYREF
  __int64 *v72; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v73; // [rsp+48h] [rbp-C8h]
  _BYTE v74[64]; // [rsp+50h] [rbp-C0h] BYREF
  _QWORD *v75; // [rsp+90h] [rbp-80h] BYREF
  __int64 v76; // [rsp+98h] [rbp-78h]
  _QWORD v77[14]; // [rsp+A0h] [rbp-70h] BYREF

  v2 = *(_QWORD *)(a2 + 328);
  v72 = (__int64 *)v74;
  v73 = 0x800000000LL;
  v70 = a2 + 320;
  if ( v2 != a2 + 320 )
  {
    v3 = a1;
    v68 = a1 + 968;
    while ( 1 )
    {
      v11 = *(_DWORD *)(v3 + 992);
      if ( !v11 )
        break;
      v4 = v11 - 1;
      v5 = 1;
      v6 = *(_QWORD *)(v3 + 976);
      v7 = 0;
      LODWORD(v8) = v4 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v9 = (__int64 *)(v6 + 16LL * (unsigned int)v8);
      v10 = *v9;
      if ( v2 != *v9 )
      {
        while ( v10 != -4096 )
        {
          if ( v7 || v10 != -8192 )
            v9 = v7;
          v8 = (unsigned int)v4 & ((_DWORD)v8 + v5);
          v10 = *(_QWORD *)(v6 + 16 * v8);
          if ( v10 == v2 )
            goto LABEL_4;
          ++v5;
          v7 = v9;
          v9 = (__int64 *)(v6 + 16 * v8);
        }
        if ( !v7 )
          v7 = v9;
        v43 = *(_DWORD *)(v3 + 984);
        ++*(_QWORD *)(v3 + 968);
        v13 = v43 + 1;
        if ( 4 * (v43 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(v3 + 988) - v13 <= v11 >> 3 )
          {
            sub_35793B0(v68, v11);
            v60 = *(_DWORD *)(v3 + 992);
            if ( !v60 )
            {
LABEL_111:
              ++*(_DWORD *)(v3 + 984);
              BUG();
            }
            v61 = v60 - 1;
            v62 = 1;
            v4 = *(_QWORD *)(v3 + 976);
            v63 = v61 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
            v13 = *(_DWORD *)(v3 + 984) + 1;
            v64 = 0;
            v7 = (__int64 *)(v4 + 16LL * v63);
            v6 = *v7;
            if ( v2 != *v7 )
            {
              while ( v6 != -4096 )
              {
                if ( v6 == -8192 && !v64 )
                  v64 = v7;
                v63 = v61 & (v62 + v63);
                v7 = (__int64 *)(v4 + 16LL * v63);
                v6 = *v7;
                if ( *v7 == v2 )
                  goto LABEL_56;
                ++v62;
              }
              if ( v64 )
                v7 = v64;
            }
          }
          goto LABEL_56;
        }
LABEL_7:
        sub_35793B0(v68, 2 * v11);
        v12 = *(_DWORD *)(v3 + 992);
        if ( !v12 )
          goto LABEL_111;
        v6 = (unsigned int)(v12 - 1);
        v4 = *(_QWORD *)(v3 + 976);
        v13 = *(_DWORD *)(v3 + 984) + 1;
        v14 = v6 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
        v7 = (__int64 *)(v4 + 16LL * v14);
        v15 = *v7;
        if ( v2 != *v7 )
        {
          v16 = 1;
          v17 = 0;
          while ( v15 != -4096 )
          {
            if ( !v17 && v15 == -8192 )
              v17 = v7;
            v14 = v6 & (v16 + v14);
            v7 = (__int64 *)(v4 + 16LL * v14);
            v15 = *v7;
            if ( *v7 == v2 )
              goto LABEL_56;
            ++v16;
          }
          if ( v17 )
            v7 = v17;
        }
LABEL_56:
        *(_DWORD *)(v3 + 984) = v13;
        if ( *v7 != -4096 )
          --*(_DWORD *)(v3 + 988);
        *v7 = v2;
        v7[1] = v2;
        v44 = *(_QWORD *)(v3 + 1000);
        LODWORD(v73) = 0;
        if ( v2 )
        {
          v45 = (unsigned int)(*(_DWORD *)(v2 + 24) + 1);
          v46 = *(_DWORD *)(v2 + 24) + 1;
        }
        else
        {
          v45 = 0;
          v46 = 0;
        }
        if ( v46 < *(_DWORD *)(v44 + 32) && *(_QWORD *)(*(_QWORD *)(v44 + 24) + 8 * v45) )
        {
          v77[0] = *(_QWORD *)(*(_QWORD *)(v44 + 24) + 8 * v45);
          v47 = 1;
          v75 = v77;
          v48 = v77;
          v69 = v3;
          v76 = 0x800000001LL;
          for ( i = 0; ; i = (unsigned int)v73 )
          {
            v50 = v48[v47 - 1];
            LODWORD(v76) = v47 - 1;
            v51 = *(_QWORD *)v50;
            if ( i + 1 > (unsigned __int64)HIDWORD(v73) )
            {
              sub_C8D5F0((__int64)&v72, v74, i + 1, 8u, v6, v4);
              i = (unsigned int)v73;
            }
            v72[i] = v51;
            v52 = *(_QWORD *)(v50 + 24);
            v53 = *(unsigned int *)(v50 + 32);
            v54 = 8 * v53;
            v55 = (unsigned int)v76;
            LODWORD(v73) = v73 + 1;
            v56 = v53 + (unsigned int)v76;
            if ( v56 > HIDWORD(v76) )
            {
              sub_C8D5F0((__int64)&v75, v77, v56, 8u, v6, v4);
              v55 = (unsigned int)v76;
            }
            v48 = v75;
            v57 = &v75[v55];
            if ( v54 )
            {
              for ( j = 0; j != v54; j += 8LL )
                v57[j / 8] = *(_QWORD *)(v52 + j);
              v48 = v75;
              LODWORD(v55) = v76;
            }
            LODWORD(v76) = v53 + v55;
            v47 = v53 + v55;
            if ( !((_DWORD)v53 + (_DWORD)v55) )
              break;
          }
          v3 = v69;
          if ( v48 != v77 )
            _libc_free((unsigned __int64)v48);
          v59 = (unsigned int)v73;
        }
        else
        {
          v59 = 0;
        }
        sub_3588750(v3, v2, v72, v59, *(_QWORD *)(v3 + 1008));
      }
LABEL_4:
      v2 = *(_QWORD *)(v2 + 8);
      if ( v70 == v2 )
      {
        v18 = *(_QWORD *)(a2 + 328);
        if ( v70 == v18 )
        {
LABEL_14:
          if ( v72 != (__int64 *)v74 )
            _libc_free((unsigned __int64)v72);
          return;
        }
        while ( 2 )
        {
          v28 = *(_DWORD *)(v3 + 992);
          v71 = v18;
          if ( !v28 )
          {
            ++*(_QWORD *)(v3 + 968);
            goto LABEL_26;
          }
          v19 = *(_QWORD *)(v3 + 976);
          v20 = 1;
          v21 = 0;
          v22 = (v28 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v23 = (__int64 *)(v19 + 16LL * v22);
          v24 = *v23;
          if ( *v23 == v18 )
          {
LABEL_20:
            v25 = v23[1];
            v26 = v18;
          }
          else
          {
            while ( v24 != -4096 )
            {
              if ( v24 == -8192 && !v21 )
                v21 = v23;
              v22 = (v28 - 1) & (v20 + v22);
              v23 = (__int64 *)(v19 + 16LL * v22);
              v24 = *v23;
              if ( v18 == *v23 )
                goto LABEL_20;
              ++v20;
            }
            if ( !v21 )
              v21 = v23;
            v36 = *(_DWORD *)(v3 + 984);
            ++*(_QWORD *)(v3 + 968);
            v33 = v36 + 1;
            if ( 4 * (v36 + 1) >= 3 * v28 )
            {
LABEL_26:
              sub_35793B0(v68, 2 * v28);
              v29 = *(_DWORD *)(v3 + 992);
              if ( !v29 )
                goto LABEL_111;
              v30 = v71;
              v31 = v29 - 1;
              v32 = *(_QWORD *)(v3 + 976);
              v33 = *(_DWORD *)(v3 + 984) + 1;
              v34 = (v29 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
              v21 = (__int64 *)(v32 + 16LL * v34);
              v35 = *v21;
              if ( *v21 != v71 )
              {
                v65 = 1;
                v66 = 0;
                while ( v35 != -4096 )
                {
                  if ( v35 == -8192 && !v66 )
                    v66 = v21;
                  v34 = v31 & (v65 + v34);
                  v21 = (__int64 *)(v32 + 16LL * v34);
                  v35 = *v21;
                  if ( v71 == *v21 )
                    goto LABEL_28;
                  ++v65;
                }
                if ( v66 )
                  v21 = v66;
              }
            }
            else
            {
              v30 = v18;
              if ( v28 - *(_DWORD *)(v3 + 988) - v33 <= v28 >> 3 )
              {
                sub_35793B0(v68, v28);
                v37 = *(_DWORD *)(v3 + 992);
                if ( !v37 )
                  goto LABEL_111;
                v38 = v37 - 1;
                v39 = *(_QWORD *)(v3 + 976);
                v40 = 0;
                v33 = *(_DWORD *)(v3 + 984) + 1;
                v41 = 1;
                v42 = (v37 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
                v21 = (__int64 *)(v39 + 16LL * v42);
                v30 = *v21;
                if ( v71 != *v21 )
                {
                  while ( v30 != -4096 )
                  {
                    if ( !v40 && v30 == -8192 )
                      v40 = v21;
                    v42 = v38 & (v41 + v42);
                    v21 = (__int64 *)(v39 + 16LL * v42);
                    v30 = *v21;
                    if ( v71 == *v21 )
                      goto LABEL_28;
                    ++v41;
                  }
                  v30 = v71;
                  if ( v40 )
                    v21 = v40;
                }
              }
            }
LABEL_28:
            *(_DWORD *)(v3 + 984) = v33;
            if ( *v21 != -4096 )
              --*(_DWORD *)(v3 + 988);
            *v21 = v30;
            v25 = 0;
            v26 = v71;
            v21[1] = 0;
          }
          v75 = (_QWORD *)v25;
          if ( v25 != v26 )
          {
            v27 = *sub_3588500(v3 + 40, (__int64 *)&v75);
            *sub_3588500(v3 + 40, &v71) = v27;
          }
          v18 = *(_QWORD *)(v18 + 8);
          if ( v2 == v18 )
            goto LABEL_14;
          continue;
        }
      }
    }
    ++*(_QWORD *)(v3 + 968);
    goto LABEL_7;
  }
}
