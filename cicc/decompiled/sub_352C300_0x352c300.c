// Function: sub_352C300
// Address: 0x352c300
//
char __fastcall sub_352C300(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rdx
  __int64 v9; // rdi
  _QWORD *v10; // rsi
  int v11; // r8d
  __int64 v12; // r9
  __int64 v13; // r10
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // r13
  _QWORD *v19; // rdi
  _QWORD *v20; // rsi
  const char *v21; // rax
  __int64 *v22; // r9
  __int64 v23; // r8
  __int64 *v24; // r10
  unsigned int v25; // r11d
  _QWORD *v26; // rdi
  _QWORD *v27; // rsi
  bool v28; // al
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 *v31; // rdi
  int v32; // eax
  unsigned int v33; // r15d
  __int64 *v34; // rcx
  __int64 v35; // rsi
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 *v38; // r8
  int v39; // edx
  unsigned int v40; // eax
  __int64 *v41; // rsi
  __int64 v42; // rdi
  int v43; // r15d
  _BYTE *v44; // r15
  __int64 v45; // rax
  __int64 *v46; // rbx
  _BYTE *v47; // r14
  __int64 v48; // rax
  __int64 v49; // rdi
  int v50; // eax
  __int64 v51; // rsi
  int v52; // ecx
  unsigned int v53; // eax
  __int64 *v54; // rdx
  _BYTE *v55; // r14
  __int64 v56; // r12
  unsigned int v57; // esi
  __int64 v58; // r8
  __int64 v59; // r11
  unsigned int v60; // edx
  __int64 **v61; // rax
  __int64 *v62; // rdi
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 *v65; // rbx
  int v66; // ecx
  int v67; // esi
  int v68; // r9d
  int v69; // r8d
  __int64 **v70; // r9
  int v71; // ecx
  int v72; // edx
  __int64 *v73; // rcx
  int v74; // edx
  int v75; // edx
  __int64 v76; // r9
  unsigned int v77; // r10d
  __int64 **v78; // rsi
  __int64 *v79; // rdi
  int v80; // edi
  int v81; // edi
  __int64 **v82; // rsi
  unsigned int i; // edx
  __int64 *v84; // r10
  int v85; // r10d
  int v86; // edx
  _BYTE *v88; // [rsp+10h] [rbp-E0h]
  int v89; // [rsp+10h] [rbp-E0h]
  int v90; // [rsp+10h] [rbp-E0h]
  __int64 v91; // [rsp+10h] [rbp-E0h]
  __int64 v92; // [rsp+10h] [rbp-E0h]
  __int64 v93; // [rsp+18h] [rbp-D8h] BYREF
  __int64 *v94; // [rsp+28h] [rbp-C8h] BYREF
  _QWORD v95[4]; // [rsp+30h] [rbp-C0h] BYREF
  char v96; // [rsp+50h] [rbp-A0h]
  char v97; // [rsp+51h] [rbp-9Fh]
  __int64 v98[2]; // [rsp+60h] [rbp-90h] BYREF
  __int64 (__fastcall *v99)(__int64 *, __int64 *, __int64); // [rsp+70h] [rbp-80h]
  _QWORD v100[2]; // [rsp+80h] [rbp-70h] BYREF
  __int64 (__fastcall *v101)(const __m128i **, const __m128i *, int); // [rsp+90h] [rbp-60h]
  _QWORD *(__fastcall *v102)(__int64 *, __int64); // [rsp+98h] [rbp-58h]
  __int64 *v103; // [rsp+A0h] [rbp-50h]
  __int64 v104; // [rsp+A8h] [rbp-48h]
  __int64 (__fastcall *v105)(const __m128i **, const __m128i *, int); // [rsp+B0h] [rbp-40h]
  _QWORD *(__fastcall *v106)(__int64 *, __int64); // [rsp+B8h] [rbp-38h]
  _BYTE v107[48]; // [rsp+C0h] [rbp-30h] BYREF

  v8 = *(_QWORD *)(a3 + 24);
  v9 = *a1;
  v93 = a2;
  if ( !(unsigned __int8)sub_2E6D360(v9, *(_QWORD *)(a2 + 24), v8) )
  {
    v88 = (_BYTE *)a1[1];
    sub_2EE7320(v98, (__int64)(v88 + 144), a2);
    sub_2EE7320(v100, a1[1] + 144, a3);
    v97 = 1;
    v21 = "Convergence control token must dominate all its uses.";
    goto LABEL_12;
  }
  v10 = (_QWORD *)(*(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8));
  if ( v10 == sub_352ADF0(*(_QWORD **)a4, (__int64)v10, &v93) )
  {
    v88 = (_BYTE *)a1[1];
    sub_2EE7320(v98, (__int64)(v88 + 144), a2);
    sub_2EE7320(v100, a1[1] + 144, a3);
    v97 = 1;
    v21 = "Convergence region is not well-nested.";
LABEL_12:
    v95[0] = v21;
    v96 = 3;
    sub_352B2E0(v88, (__int64)v95, v98, 2);
    if ( !v101 )
      goto LABEL_14;
    goto LABEL_13;
  }
  v14 = v11 - 1;
  if ( a2 != *(_QWORD *)(v12 + v13 - 8) )
  {
    do
    {
      v15 = v14;
      *(_DWORD *)(a4 + 8) = v14--;
    }
    while ( *(_QWORD *)(v12 + 8 * v15 - 8) != a2 );
  }
  v16 = *(_QWORD *)(a3 + 24);
  v17 = sub_2E5E6D0(a1[1] + 48, v16);
  v94 = (__int64 *)v17;
  if ( v17 )
  {
    v18 = *(_QWORD *)(a2 + 24);
    if ( v16 != v18 )
    {
      v98[0] = *(_QWORD *)(a2 + 24);
      if ( *(_DWORD *)(v17 + 72) )
      {
        v36 = *(_QWORD *)(v17 + 64);
        v37 = *(unsigned int *)(v17 + 80);
        v38 = (__int64 *)(v36 + 8 * v37);
        if ( !(_DWORD)v37 )
          goto LABEL_16;
        v39 = v37 - 1;
        v40 = v39 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v41 = (__int64 *)(v36 + 8LL * v40);
        v42 = *v41;
        if ( v18 != *v41 )
        {
          v67 = 1;
          while ( v42 != -4096 )
          {
            v68 = v67 + 1;
            v40 = v39 & (v67 + v40);
            v41 = (__int64 *)(v36 + 8LL * v40);
            v42 = *v41;
            if ( v18 == *v41 )
              goto LABEL_30;
            v67 = v68;
          }
          goto LABEL_16;
        }
LABEL_30:
        LOBYTE(v17) = v41 != v38;
      }
      else
      {
        v19 = *(_QWORD **)(v17 + 88);
        v20 = &v19[*(unsigned int *)(v17 + 96)];
        LOBYTE(v17) = v20 != sub_352AD30(v19, (__int64)v20, v98);
      }
      if ( (_BYTE)v17 )
        return v17;
LABEL_16:
      if ( (unsigned int)sub_352B010(a3) == 2 )
      {
        v22 = v94;
        v23 = *v94;
        if ( *v94 )
        {
          v24 = v98;
          v25 = ((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4);
          while ( 1 )
          {
            v98[0] = v18;
            if ( *(_DWORD *)(v23 + 72) )
            {
              v29 = *(unsigned int *)(v23 + 80);
              v30 = *(_QWORD *)(v23 + 64);
              v31 = (__int64 *)(v30 + 8 * v29);
              if ( !(_DWORD)v29 )
                goto LABEL_21;
              v32 = v29 - 1;
              v33 = (v29 - 1) & v25;
              v34 = (__int64 *)(v30 + 8LL * v33);
              v35 = *v34;
              if ( v18 != *v34 )
              {
                v66 = 1;
                while ( v35 != -4096 )
                {
                  v33 = v32 & (v66 + v33);
                  v89 = v66 + 1;
                  v34 = (__int64 *)(v30 + 8LL * v33);
                  v35 = *v34;
                  if ( v18 == *v34 )
                    goto LABEL_26;
                  v66 = v89;
                }
                goto LABEL_21;
              }
LABEL_26:
              v28 = v31 != v34;
            }
            else
            {
              v26 = *(_QWORD **)(v23 + 88);
              v27 = &v26[*(unsigned int *)(v23 + 96)];
              v28 = v27 != sub_352AD30(v26, (__int64)v27, v24);
            }
            if ( v28 )
              break;
LABEL_21:
            v94 = (__int64 *)v23;
            v22 = (__int64 *)v23;
            if ( !*(_QWORD *)v23 )
              break;
            v23 = *(_QWORD *)v23;
          }
        }
        v43 = *((_DWORD *)v22 + 4);
        if ( v43 != 1 || v16 != *(_QWORD *)v22[1] )
        {
          v44 = (_BYTE *)a1[1];
          sub_2EE7320(v98, (__int64)(v44 + 144), a3);
          sub_2EE7370(v100, a1[1] + 144, v16);
          v45 = a1[1];
          v46 = (__int64 *)v107;
          v97 = 1;
          v103 = v94;
          v104 = v45 + 48;
          v96 = 3;
          v105 = sub_2E5D7C0;
          v106 = sub_2E5F810;
          v95[0] = "Cycle heart must dominate all blocks in the cycle.";
          sub_352B2E0(v44, (__int64)v95, v98, 3);
          do
          {
            v17 = *(v46 - 2);
            v46 -= 4;
            if ( v17 )
              LOBYTE(v17) = ((__int64 (__fastcall *)(__int64 *, __int64 *, __int64))v17)(v46, v46, 3);
          }
          while ( v46 != v98 );
          return v17;
        }
        v49 = a1[2];
        v50 = *(_DWORD *)(v49 + 24);
        v51 = *(_QWORD *)(v49 + 8);
        if ( v50 )
        {
          v52 = v50 - 1;
          v53 = (v50 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v54 = *(__int64 **)(v51 + 16LL * v53);
          if ( v54 == v22 )
          {
LABEL_42:
            v55 = (_BYTE *)a1[1];
            sub_2EE7320(v98, (__int64)(v55 + 144), a3);
            v56 = a1[2];
            v57 = *(_DWORD *)(v56 + 24);
            v58 = a1[1] + 144;
            if ( v57 )
            {
              v59 = *(_QWORD *)(v56 + 8);
              v60 = (v57 - 1) & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4));
              v61 = (__int64 **)(v59 + 16LL * v60);
              v62 = *v61;
              if ( v94 == *v61 )
              {
LABEL_44:
                v63 = (__int64)v61[1];
LABEL_45:
                sub_2EE7320(v100, v58, v63);
                v64 = a1[1];
                v65 = (__int64 *)v107;
                v97 = 1;
                v103 = v94;
                v104 = v64 + 48;
                v96 = 3;
                v105 = sub_2E5D7C0;
                v106 = sub_2E5F810;
                v95[0] = "Two static convergence token uses in a cycle that does not contain either token's definition.";
                sub_352B2E0(v55, (__int64)v95, v98, 3);
                do
                {
                  v17 = *(v65 - 2);
                  v65 -= 4;
                  if ( v17 )
                    LOBYTE(v17) = ((__int64 (__fastcall *)(__int64 *, __int64 *, __int64))v17)(v65, v65, 3);
                }
                while ( v65 != v98 );
                return v17;
              }
              v90 = 1;
              v70 = 0;
              while ( v62 != (__int64 *)-4096LL )
              {
                if ( !v70 && v62 == (__int64 *)-8192LL )
                  v70 = v61;
                v60 = (v57 - 1) & (v90 + v60);
                v61 = (__int64 **)(v59 + 16LL * v60);
                v62 = *v61;
                if ( v94 == *v61 )
                  goto LABEL_44;
                ++v90;
              }
              v71 = *(_DWORD *)(v56 + 16);
              if ( v70 )
                v61 = v70;
              ++*(_QWORD *)v56;
              v72 = v71 + 1;
              if ( 4 * (v71 + 1) < 3 * v57 )
              {
                v73 = v94;
                if ( v57 - *(_DWORD *)(v56 + 20) - v72 > v57 >> 3 )
                {
LABEL_69:
                  *(_DWORD *)(v56 + 16) = v72;
                  if ( *v61 != (__int64 *)-4096LL )
                    --*(_DWORD *)(v56 + 20);
                  *v61 = v73;
                  v63 = 0;
                  v61[1] = 0;
                  goto LABEL_45;
                }
                v91 = v58;
                sub_352BED0(v56, v57);
                v74 = *(_DWORD *)(v56 + 24);
                if ( v74 )
                {
                  v73 = v94;
                  v75 = v74 - 1;
                  v76 = *(_QWORD *)(v56 + 8);
                  v58 = v91;
                  v77 = v75 & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4));
                  v61 = 0;
                  while ( 1 )
                  {
                    v78 = (__int64 **)(v76 + 16LL * v77);
                    v79 = *v78;
                    if ( v94 == *v78 )
                    {
                      v72 = *(_DWORD *)(v56 + 16) + 1;
                      v61 = (__int64 **)(v76 + 16LL * v77);
                      goto LABEL_69;
                    }
                    if ( v79 == (__int64 *)-4096LL )
                      break;
                    if ( v79 != (__int64 *)-8192LL || v61 )
                      v78 = v61;
                    v85 = v43 + v77;
                    v61 = v78;
                    ++v43;
                    v77 = v75 & v85;
                  }
                  if ( !v61 )
                    v61 = (__int64 **)(v76 + 16LL * v77);
                  v72 = *(_DWORD *)(v56 + 16) + 1;
                  goto LABEL_69;
                }
LABEL_102:
                ++*(_DWORD *)(v56 + 16);
                BUG();
              }
            }
            else
            {
              ++*(_QWORD *)v56;
            }
            v92 = v58;
            sub_352BED0(v56, 2 * v57);
            v80 = *(_DWORD *)(v56 + 24);
            if ( v80 )
            {
              v73 = v94;
              v81 = v80 - 1;
              v82 = 0;
              v58 = v92;
              for ( i = v81 & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4)); ; i = v81 & v86 )
              {
                v61 = (__int64 **)(*(_QWORD *)(v56 + 8) + 16LL * i);
                v84 = *v61;
                if ( v94 == *v61 )
                {
                  v72 = *(_DWORD *)(v56 + 16) + 1;
                  goto LABEL_69;
                }
                if ( v84 == (__int64 *)-4096LL )
                  break;
                if ( v82 || v84 != (__int64 *)-8192LL )
                  v61 = v82;
                v86 = v43 + i;
                v82 = v61;
                ++v43;
              }
              if ( v82 )
                v61 = v82;
              v72 = *(_DWORD *)(v56 + 16) + 1;
              goto LABEL_69;
            }
            goto LABEL_102;
          }
          v69 = 1;
          while ( v54 != (__int64 *)-4096LL )
          {
            v53 = v52 & (v69 + v53);
            v54 = *(__int64 **)(v51 + 16LL * v53);
            if ( v54 == v22 )
              goto LABEL_42;
            ++v69;
          }
        }
        v17 = (__int64)sub_352C0B0(v49, (__int64 *)&v94);
        *(_QWORD *)v17 = a3;
        return v17;
      }
      v47 = (_BYTE *)a1[1];
      sub_2EE7320(v98, (__int64)(v47 + 144), a3);
      v48 = a1[1];
      v97 = 1;
      v96 = 3;
      v100[1] = v48 + 48;
      v100[0] = v94;
      v101 = sub_2E5D7C0;
      v102 = sub_2E5F810;
      v95[0] = "Convergence token used by an instruction other than llvm.experimental.convergence.loop in a cycle that do"
               "es not contain the token's definition.";
      sub_352B2E0(v47, (__int64)v95, v98, 2);
      if ( !v101 )
        goto LABEL_14;
LABEL_13:
      v101((const __m128i **)v100, (const __m128i *)v100, 3);
LABEL_14:
      LOBYTE(v17) = (_BYTE)v99;
      if ( v99 )
        LOBYTE(v17) = v99(v98, v98, 3);
    }
  }
  return v17;
}
