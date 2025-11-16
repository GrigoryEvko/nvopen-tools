// Function: sub_1942B80
// Address: 0x1942b80
//
void __fastcall sub_1942B80(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // r15
  int v5; // r13d
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 *v8; // rbx
  __int64 v9; // rbx
  __int64 v10; // rdx
  const void *v11; // rdi
  const void *v12; // rsi
  __int64 v13; // rcx
  int v14; // r10d
  unsigned __int64 v15; // r8
  unsigned __int64 v16; // r8
  unsigned int i; // eax
  __int64 v18; // r12
  unsigned int v19; // eax
  unsigned int v20; // esi
  __int64 v21; // rdi
  __int64 v22; // rdx
  int v23; // r12d
  __int64 *v24; // r11
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int64 v27; // r9
  unsigned __int64 v28; // r9
  unsigned int j; // eax
  __int64 *v30; // r9
  const void *v31; // r13
  unsigned int v32; // eax
  __int64 v33; // rdi
  __int64 v34; // rdi
  unsigned int v35; // eax
  const void *v36; // rdi
  int v37; // eax
  int v38; // ecx
  int v39; // edi
  int v40; // edi
  __int64 *v41; // r10
  __int64 v42; // rsi
  int v43; // r11d
  __int64 v44; // r8
  unsigned __int64 v45; // r8
  unsigned __int64 v46; // r8
  unsigned int m; // eax
  const void *v48; // r8
  unsigned int v49; // eax
  int v50; // edi
  int v51; // edi
  int v52; // r11d
  __int64 v53; // rsi
  __int64 v54; // r8
  unsigned __int64 v55; // r8
  unsigned __int64 v56; // r8
  unsigned int k; // eax
  const void **v58; // r8
  const void *v59; // r10
  unsigned int v60; // eax
  __int64 v61; // [rsp-E8h] [rbp-E8h] BYREF
  unsigned int v62; // [rsp-E0h] [rbp-E0h]
  __int64 v63; // [rsp-D8h] [rbp-D8h] BYREF
  unsigned int v64; // [rsp-D0h] [rbp-D0h]
  __int64 v65; // [rsp-C8h] [rbp-C8h] BYREF
  unsigned int v66; // [rsp-C0h] [rbp-C0h]
  __int64 v67; // [rsp-B8h] [rbp-B8h]
  unsigned int v68; // [rsp-B0h] [rbp-B0h]
  const void *v69; // [rsp-A8h] [rbp-A8h] BYREF
  unsigned int v70; // [rsp-A0h] [rbp-A0h]
  __int64 v71; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v72; // [rsp-90h] [rbp-90h]
  const void *v73; // [rsp-88h] [rbp-88h] BYREF
  unsigned int v74; // [rsp-80h] [rbp-80h]
  __int64 v75; // [rsp-78h] [rbp-78h] BYREF
  unsigned int v76; // [rsp-70h] [rbp-70h]
  const void *v77; // [rsp-68h] [rbp-68h] BYREF
  const void *v78; // [rsp-60h] [rbp-60h]
  const void *v79; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v80; // [rsp-50h] [rbp-50h]
  __int64 v81; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v82; // [rsp-40h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) == 75 && **(_QWORD **)a1 == *(_QWORD *)(a2 - 48) )
  {
    v4 = *(_QWORD *)(a2 - 24);
    if ( v4 )
    {
      v5 = *(_WORD *)(a2 + 18) & 0x7FFF;
      if ( !a3 )
        v5 = sub_15FF0F0(v5);
      v6 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
      v7 = sub_146F1B0(v6, v4);
      v8 = sub_1477920(v6, v7, 1u);
      v62 = *((_DWORD *)v8 + 2);
      if ( v62 > 0x40 )
        sub_16A4FD0((__int64)&v61, (const void **)v8);
      else
        v61 = *v8;
      v64 = *((_DWORD *)v8 + 6);
      if ( v64 > 0x40 )
        sub_16A4FD0((__int64)&v63, (const void **)v8 + 2);
      else
        v63 = v8[2];
      sub_158AE10((__int64)&v65, v5, (__int64)&v61);
      sub_1592D00((__int64)&v69, (__int64)&v65, **(_QWORD **)(a1 + 16));
      v9 = *(_QWORD *)(a1 + 8);
      v74 = v70;
      if ( v70 > 0x40 )
        sub_16A4FD0((__int64)&v73, &v69);
      else
        v73 = v69;
      v76 = v72;
      if ( v72 > 0x40 )
        sub_16A4FD0((__int64)&v75, (const void **)&v71);
      else
        v75 = v71;
      v10 = *(unsigned int *)(v9 + 584);
      v11 = **(const void ***)(a1 + 32);
      v12 = **(const void ***)(a1 + 24);
      if ( !(_DWORD)v10 )
        goto LABEL_26;
      v13 = *(_QWORD *)(v9 + 568);
      v14 = 1;
      v15 = (((((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
             | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32)) >> 22)
          ^ ((((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
            | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32));
      v16 = ((9 * (((v15 - 1 - (v15 << 13)) >> 8) ^ (v15 - 1 - (v15 << 13)))) >> 15)
          ^ (9 * (((v15 - 1 - (v15 << 13)) >> 8) ^ (v15 - 1 - (v15 << 13))));
      for ( i = (v10 - 1) & (((v16 - 1 - (v16 << 27)) >> 31) ^ (v16 - 1 - ((_DWORD)v16 << 27))); ; i = (v10 - 1) & v19 )
      {
        v18 = v13 + 48LL * i;
        if ( v12 == *(const void **)v18 && v11 == *(const void **)(v18 + 8) )
          break;
        if ( *(_QWORD *)v18 == -8 && *(_QWORD *)(v18 + 8) == -8 )
          goto LABEL_26;
        v19 = v14 + i;
        ++v14;
      }
      if ( v18 == 48 * v10 + v13 )
      {
LABEL_26:
        v77 = v12;
        v78 = v11;
        v80 = v74;
        if ( v74 > 0x40 )
          sub_16A4FD0((__int64)&v79, &v73);
        else
          v79 = v73;
        v82 = v76;
        if ( v76 > 0x40 )
          sub_16A4FD0((__int64)&v81, (const void **)&v75);
        else
          v81 = v75;
        v20 = *(_DWORD *)(v9 + 584);
        v21 = v9 + 560;
        if ( v20 )
        {
          v22 = (__int64)v77;
          v23 = 1;
          v24 = 0;
          v25 = *(_QWORD *)(v9 + 568);
          v26 = ((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4);
          v27 = (((v26 | ((unsigned __int64)(((unsigned int)v77 >> 9) ^ ((unsigned int)v22 >> 4)) << 32))
                - 1
                - (v26 << 32)) >> 22)
              ^ ((v26 | ((unsigned __int64)(((unsigned int)v77 >> 9) ^ ((unsigned int)v22 >> 4)) << 32))
               - 1
               - (v26 << 32));
          v28 = ((9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13)))) >> 15)
              ^ (9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13))));
          for ( j = (v20 - 1) & (((v28 - 1 - (v28 << 27)) >> 31) ^ (v28 - 1 - ((_DWORD)v28 << 27))); ; j = (v20 - 1) & v32 )
          {
            v30 = (__int64 *)(v25 + 48LL * j);
            v31 = (const void *)*v30;
            if ( (const void *)*v30 == v77 && (const void *)v30[1] == v78 )
            {
              if ( v82 > 0x40 && v81 )
                j_j___libc_free_0_0(v81);
              goto LABEL_84;
            }
            if ( v31 == (const void *)-8LL )
            {
              if ( v30[1] == -8 )
              {
                v37 = *(_DWORD *)(v9 + 576);
                if ( v24 )
                  v30 = v24;
                ++*(_QWORD *)(v9 + 560);
                v38 = v37 + 1;
                if ( 4 * (v37 + 1) < 3 * v20 )
                {
                  if ( v20 - *(_DWORD *)(v9 + 580) - v38 > v20 >> 3 )
                    goto LABEL_81;
                  sub_1942860(v21, v20);
                  v50 = *(_DWORD *)(v9 + 584);
                  if ( v50 )
                  {
                    v22 = (__int64)v77;
                    v51 = v50 - 1;
                    v52 = 1;
                    v53 = *(_QWORD *)(v9 + 568);
                    v54 = ((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4);
                    v30 = 0;
                    v55 = (((v54 | ((unsigned __int64)(((unsigned int)v77 >> 9) ^ ((unsigned int)v22 >> 4)) << 32))
                          - 1
                          - (v54 << 32)) >> 22)
                        ^ ((v54 | ((unsigned __int64)(((unsigned int)v77 >> 9) ^ ((unsigned int)v22 >> 4)) << 32))
                         - 1
                         - (v54 << 32));
                    v56 = ((9 * (((v55 - 1 - (v55 << 13)) >> 8) ^ (v55 - 1 - (v55 << 13)))) >> 15)
                        ^ (9 * (((v55 - 1 - (v55 << 13)) >> 8) ^ (v55 - 1 - (v55 << 13))));
                    for ( k = v51 & (((v56 - 1 - (v56 << 27)) >> 31) ^ (v56 - 1 - ((_DWORD)v56 << 27))); ; k = v51 & v60 )
                    {
                      v58 = (const void **)(v53 + 48LL * k);
                      v59 = *v58;
                      if ( *v58 == v77 && v58[1] == v78 )
                      {
                        v30 = (__int64 *)(v53 + 48LL * k);
                        v38 = *(_DWORD *)(v9 + 576) + 1;
                        goto LABEL_81;
                      }
                      if ( v59 == (const void *)-8LL )
                      {
                        if ( v58[1] == (const void *)-8LL )
                        {
                          if ( !v30 )
                            v30 = (__int64 *)(v53 + 48LL * k);
                          v38 = *(_DWORD *)(v9 + 576) + 1;
                          goto LABEL_81;
                        }
                      }
                      else if ( v59 == (const void *)-16LL && v58[1] == (const void *)-16LL && !v30 )
                      {
                        v30 = (__int64 *)(v53 + 48LL * k);
                      }
                      v60 = v52 + k;
                      ++v52;
                    }
                  }
                  goto LABEL_124;
                }
LABEL_94:
                sub_1942860(v21, 2 * v20);
                v39 = *(_DWORD *)(v9 + 584);
                if ( v39 )
                {
                  v22 = (__int64)v77;
                  v40 = v39 - 1;
                  v41 = 0;
                  v42 = *(_QWORD *)(v9 + 568);
                  v43 = 1;
                  v44 = ((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4);
                  v45 = (((v44 | ((unsigned __int64)(((unsigned int)v77 >> 9) ^ ((unsigned int)v22 >> 4)) << 32))
                        - 1
                        - (v44 << 32)) >> 22)
                      ^ ((v44 | ((unsigned __int64)(((unsigned int)v77 >> 9) ^ ((unsigned int)v22 >> 4)) << 32))
                       - 1
                       - (v44 << 32));
                  v46 = ((9 * (((v45 - 1 - (v45 << 13)) >> 8) ^ (v45 - 1 - (v45 << 13)))) >> 15)
                      ^ (9 * (((v45 - 1 - (v45 << 13)) >> 8) ^ (v45 - 1 - (v45 << 13))));
                  for ( m = v40 & (((v46 - 1 - (v46 << 27)) >> 31) ^ (v46 - 1 - ((_DWORD)v46 << 27))); ; m = v40 & v49 )
                  {
                    v30 = (__int64 *)(v42 + 48LL * m);
                    v48 = (const void *)*v30;
                    if ( (const void *)*v30 == v77 && (const void *)v30[1] == v78 )
                    {
                      v38 = *(_DWORD *)(v9 + 576) + 1;
                      goto LABEL_81;
                    }
                    if ( v48 == (const void *)-8LL )
                    {
                      if ( v30[1] == -8 )
                      {
                        if ( v41 )
                          v30 = v41;
                        v38 = *(_DWORD *)(v9 + 576) + 1;
LABEL_81:
                        *(_DWORD *)(v9 + 576) = v38;
                        if ( *v30 != -8 || v30[1] != -8 )
                          --*(_DWORD *)(v9 + 580);
                        *v30 = v22;
                        v30[1] = (__int64)v78;
                        *((_DWORD *)v30 + 6) = v80;
                        v30[2] = (__int64)v79;
                        v80 = 0;
                        *((_DWORD *)v30 + 10) = v82;
                        v30[4] = v81;
LABEL_84:
                        if ( v80 <= 0x40 )
                          goto LABEL_49;
                        v36 = v79;
                        if ( !v79 )
                          goto LABEL_49;
LABEL_48:
                        j_j___libc_free_0_0(v36);
                        goto LABEL_49;
                      }
                    }
                    else if ( v48 == (const void *)-16LL && v30[1] == -16 && !v41 )
                    {
                      v41 = (__int64 *)(v42 + 48LL * m);
                    }
                    v49 = v43 + m;
                    ++v43;
                  }
                }
LABEL_124:
                ++*(_DWORD *)(v9 + 576);
                BUG();
              }
            }
            else if ( v31 == (const void *)-16LL && v30[1] == -16 && !v24 )
            {
              v24 = (__int64 *)(v25 + 48LL * j);
            }
            v32 = v23 + j;
            ++v23;
          }
        }
        ++*(_QWORD *)(v9 + 560);
        goto LABEL_94;
      }
      sub_158BE00((__int64)&v77, (__int64)&v73, v18 + 16);
      if ( *(_DWORD *)(v18 + 24) > 0x40u )
      {
        v33 = *(_QWORD *)(v18 + 16);
        if ( v33 )
          j_j___libc_free_0_0(v33);
      }
      *(_QWORD *)(v18 + 16) = v77;
      *(_DWORD *)(v18 + 24) = (_DWORD)v78;
      LODWORD(v78) = 0;
      if ( *(_DWORD *)(v18 + 40) <= 0x40u || (v34 = *(_QWORD *)(v18 + 32)) == 0 )
      {
        *(_QWORD *)(v18 + 32) = v79;
        *(_DWORD *)(v18 + 40) = v80;
        goto LABEL_49;
      }
      j_j___libc_free_0_0(v34);
      v35 = (unsigned int)v78;
      *(_QWORD *)(v18 + 32) = v79;
      *(_DWORD *)(v18 + 40) = v80;
      if ( v35 > 0x40 )
      {
        v36 = v77;
        if ( v77 )
          goto LABEL_48;
      }
LABEL_49:
      if ( v76 > 0x40 && v75 )
        j_j___libc_free_0_0(v75);
      if ( v74 > 0x40 && v73 )
        j_j___libc_free_0_0(v73);
      if ( v72 > 0x40 && v71 )
        j_j___libc_free_0_0(v71);
      if ( v70 > 0x40 && v69 )
        j_j___libc_free_0_0(v69);
      if ( v68 > 0x40 && v67 )
        j_j___libc_free_0_0(v67);
      if ( v66 > 0x40 && v65 )
        j_j___libc_free_0_0(v65);
      if ( v64 > 0x40 && v63 )
        j_j___libc_free_0_0(v63);
      if ( v62 > 0x40 && v61 )
        j_j___libc_free_0_0(v61);
    }
  }
}
