// Function: sub_22B65C0
// Address: 0x22b65c0
//
__int64 __fastcall sub_22B65C0(__int64 a1, __int64 a2, __int64 **a3, __int64 a4)
{
  __int64 v6; // rbx
  unsigned int v7; // ecx
  __int64 v8; // rsi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  int v12; // eax
  __int64 v13; // rax
  void *v14; // rax
  const void *v15; // rsi
  unsigned int v16; // esi
  int v17; // eax
  __int64 v18; // rdi
  int *v19; // r9
  int v20; // r10d
  unsigned int v21; // edx
  int *v22; // r13
  int v23; // ecx
  __int64 v24; // rdi
  __int64 v25; // rsi
  int v27; // edx
  int *v28; // rbx
  int *v29; // r12
  __int64 v30; // rax
  __int64 v31; // rcx
  int v32; // edx
  unsigned int v33; // r9d
  int *v34; // rsi
  int v35; // r8d
  int v36; // eax
  int v37; // edx
  __int64 v38; // rax
  unsigned int v39; // edi
  int *v40; // rcx
  int v41; // eax
  int v42; // r11d
  int *v43; // r10
  int v44; // edx
  int v45; // esi
  int v46; // r11d
  __int64 v47; // rsi
  int v48; // eax
  _DWORD *v49; // rax
  _DWORD *v50; // rdx
  int v51; // r8d
  _DWORD *v52; // rcx
  __int64 *v53; // rdx
  __int64 v54; // r12
  __int64 v55; // r10
  int v56; // r15d
  unsigned int v57; // esi
  __int64 *v58; // rax
  __int64 v59; // rbx
  __int64 v60; // rcx
  int v61; // esi
  __int64 v62; // r11
  unsigned int v63; // edi
  int *v64; // rax
  int v65; // ebx
  int v66; // ecx
  __int64 v67; // rdi
  int v68; // ecx
  int v69; // ebx
  int *v70; // rsi
  int v71; // r11d
  int v72; // edi
  int v73; // ecx
  __int64 v74; // rcx
  __int64 v75; // rdi
  __int64 v76; // r11
  int v77; // esi
  int v78; // eax
  int v79; // eax
  unsigned int v80; // edx
  int v81; // esi
  int v82; // edi
  int v83; // r13d
  __int64 v84; // rsi
  int v85; // r8d
  int v86; // ecx
  int *v87; // rax
  __int64 v89; // [rsp+18h] [rbp-98h]
  int v90; // [rsp+20h] [rbp-90h]
  int v91; // [rsp+20h] [rbp-90h]
  __int64 v93; // [rsp+30h] [rbp-80h]
  __int64 v94; // [rsp+30h] [rbp-80h]
  __int64 *v95; // [rsp+38h] [rbp-78h]
  int *v96; // [rsp+48h] [rbp-68h] BYREF
  __int64 v97; // [rsp+50h] [rbp-60h] BYREF
  __int64 v98; // [rsp+58h] [rbp-58h]
  __int64 v99; // [rsp+60h] [rbp-50h]
  __int64 v100; // [rsp+68h] [rbp-48h]
  unsigned int v101; // [rsp+70h] [rbp-40h]

  v95 = *a3;
  v89 = (__int64)&(*a3)[(_QWORD)a3[1]];
  if ( *a3 != (__int64 *)v89 )
  {
    do
    {
      v6 = *v95;
      v7 = *(_DWORD *)(a1 + 24);
      v8 = *(_QWORD *)(a1 + 8);
      v93 = *v95;
      if ( v7 )
      {
        v9 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( v6 == *v10 )
          goto LABEL_4;
        v48 = 1;
        while ( v11 != -4096 )
        {
          v85 = v48 + 1;
          v9 = (v7 - 1) & (v48 + v9);
          v10 = (__int64 *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( v93 == *v10 )
            goto LABEL_4;
          v48 = v85;
        }
      }
      v10 = (__int64 *)(v8 + 16LL * v7);
LABEL_4:
      v12 = *((_DWORD *)v10 + 2);
      v98 = 0;
      LODWORD(v97) = v12;
      v99 = 0;
      v100 = 0;
      v101 = 0;
      sub_C7D6A0(0, 0, 4);
      v13 = *(unsigned int *)(a4 + 24);
      v101 = v13;
      if ( (_DWORD)v13 )
      {
        v14 = (void *)sub_C7D670(4 * v13, 4);
        v15 = *(const void **)(a4 + 8);
        v99 = (__int64)v14;
        v100 = *(_QWORD *)(a4 + 16);
        memcpy(v14, v15, 4LL * v101);
        v16 = *(_DWORD *)(a2 + 24);
        if ( !v16 )
          goto LABEL_11;
      }
      else
      {
        v16 = *(_DWORD *)(a2 + 24);
        v99 = 0;
        v100 = 0;
        if ( !v16 )
        {
LABEL_11:
          ++*(_QWORD *)a2;
          v96 = 0;
          goto LABEL_12;
        }
      }
      v17 = v97;
      v18 = *(_QWORD *)(a2 + 8);
      v19 = 0;
      v20 = 1;
      v21 = (v16 - 1) & (37 * v97);
      v22 = (int *)(v18 + 40LL * v21);
      v23 = *v22;
      if ( (_DWORD)v97 != *v22 )
      {
        while ( v23 != -1 )
        {
          if ( v23 == -2 && !v19 )
            v19 = v22;
          v21 = (v16 - 1) & (v20 + v21);
          v22 = (int *)(v18 + 40LL * v21);
          v23 = *v22;
          if ( (_DWORD)v97 == *v22 )
            goto LABEL_7;
          ++v20;
        }
        v82 = *(_DWORD *)(a2 + 16);
        if ( v19 )
          v22 = v19;
        ++*(_QWORD *)a2;
        v27 = v82 + 1;
        v96 = v22;
        if ( 4 * (v82 + 1) >= 3 * v16 )
        {
LABEL_12:
          v16 *= 2;
        }
        else if ( v16 - *(_DWORD *)(a2 + 20) - v27 > v16 >> 3 )
        {
          goto LABEL_96;
        }
        sub_22B36A0(a2, v16);
        sub_22B1BB0(a2, (int *)&v97, &v96);
        v17 = v97;
        v22 = v96;
        v27 = *(_DWORD *)(a2 + 16) + 1;
LABEL_96:
        *(_DWORD *)(a2 + 16) = v27;
        if ( *v22 != -1 )
          --*(_DWORD *)(a2 + 20);
        *((_QWORD *)v22 + 3) = 0;
        v24 = 0;
        v25 = 0;
        *((_QWORD *)v22 + 2) = 0;
        v22[8] = 0;
        *v22 = v17;
        *((_QWORD *)v22 + 1) = 1;
        ++v98;
        *((_QWORD *)v22 + 2) = v99;
        *((_QWORD *)v22 + 3) = v100;
        v22[8] = v101;
        v99 = 0;
        v100 = 0;
        v101 = 0;
        goto LABEL_8;
      }
LABEL_7:
      v24 = v99;
      v25 = 4LL * v101;
LABEL_8:
      sub_C7D6A0(v24, v25, 4);
      v97 = 0;
      v98 = 0;
      v99 = 0;
      v100 = 0;
      if ( !v22[6] )
        goto LABEL_9;
      v28 = (int *)*((_QWORD *)v22 + 2);
      v29 = &v28[v22[8]];
      if ( v28 == v29 )
        goto LABEL_9;
      while ( (unsigned int)*v28 > 0xFFFFFFFD )
      {
        if ( v29 == ++v28 )
          goto LABEL_9;
      }
      if ( v29 == v28 )
      {
LABEL_9:
        sub_C7D6A0(v98, 4LL * (unsigned int)v100, 4);
        return 0;
      }
      do
      {
        v30 = *(unsigned int *)(a4 + 24);
        v31 = *(_QWORD *)(a4 + 8);
        if ( (_DWORD)v30 )
        {
          v32 = *v28;
          v33 = (v30 - 1) & (37 * *v28);
          v34 = (int *)(v31 + 4LL * v33);
          v35 = *v34;
          if ( *v34 == *v28 )
          {
LABEL_21:
            if ( v34 != (int *)(v31 + 4 * v30) )
            {
              if ( !(_DWORD)v100 )
              {
                ++v97;
                v96 = 0;
                goto LABEL_83;
              }
              v39 = (v100 - 1) & (37 * *v28);
              v40 = (int *)(v98 + 4LL * v39);
              v41 = *v40;
              if ( *v40 != v32 )
              {
                v42 = 1;
                v43 = 0;
                while ( v41 != -1 )
                {
                  if ( v41 == -2 && !v43 )
                    v43 = v40;
                  v39 = (v100 - 1) & (v42 + v39);
                  v40 = (int *)(v98 + 4LL * v39);
                  v41 = *v40;
                  if ( v32 == *v40 )
                    goto LABEL_22;
                  ++v42;
                }
                if ( !v43 )
                  v43 = v40;
                ++v97;
                v44 = v99 + 1;
                v96 = v43;
                if ( 4 * ((int)v99 + 1) < (unsigned int)(3 * v100) )
                {
                  if ( (int)v100 - HIDWORD(v99) - v44 <= (unsigned int)v100 >> 3 )
                  {
                    sub_A08C50((__int64)&v97, v100);
                    sub_22B31A0((__int64)&v97, v28, &v96);
                    v43 = v96;
                    v44 = v99 + 1;
                  }
LABEL_42:
                  LODWORD(v99) = v44;
                  if ( *v43 != -1 )
                    --HIDWORD(v99);
                  *v43 = *v28;
                  goto LABEL_22;
                }
LABEL_83:
                sub_A08C50((__int64)&v97, 2 * v100);
                if ( (_DWORD)v100 )
                {
                  v80 = (v100 - 1) & (37 * *v28);
                  v43 = (int *)(v98 + 4LL * v80);
                  v81 = *v43;
                  if ( *v43 == *v28 )
                  {
LABEL_85:
                    v96 = v43;
                    v44 = v99 + 1;
                  }
                  else
                  {
                    v86 = 1;
                    v87 = 0;
                    while ( v81 != -1 )
                    {
                      if ( !v87 && v81 == -2 )
                        v87 = v43;
                      v80 = (v100 - 1) & (v80 + v86);
                      v43 = (int *)(v98 + 4LL * v80);
                      v81 = *v43;
                      if ( *v28 == *v43 )
                        goto LABEL_85;
                      ++v86;
                    }
                    if ( !v87 )
                      v87 = v43;
                    v44 = v99 + 1;
                    v96 = v87;
                    v43 = v87;
                  }
                }
                else
                {
                  v96 = 0;
                  v43 = 0;
                  v44 = v99 + 1;
                }
                goto LABEL_42;
              }
            }
          }
          else
          {
            v45 = 1;
            while ( v35 != -1 )
            {
              v46 = v45 + 1;
              v47 = ((_DWORD)v30 - 1) & (v33 + v45);
              v33 = v47;
              v34 = (int *)(v31 + 4 * v47);
              v35 = *v34;
              if ( v32 == *v34 )
                goto LABEL_21;
              v45 = v46;
            }
          }
        }
LABEL_22:
        if ( ++v28 == v29 )
          break;
        while ( (unsigned int)*v28 > 0xFFFFFFFD )
        {
          if ( v29 == ++v28 )
            goto LABEL_25;
        }
      }
      while ( v29 != v28 );
LABEL_25:
      v36 = v99;
      if ( !(_DWORD)v99 )
        goto LABEL_9;
      v37 = v22[6];
      if ( v37 != (_DWORD)v99 )
      {
        ++*((_QWORD *)v22 + 1);
        ++v97;
        v38 = *((_QWORD *)v22 + 2);
        *((_QWORD *)v22 + 2) = v98;
        v98 = v38;
        v22[6] = v99;
        LODWORD(v99) = v37;
        LODWORD(v38) = v22[7];
        v22[7] = HIDWORD(v99);
        HIDWORD(v99) = v38;
        LODWORD(v38) = v22[8];
        v22[8] = v100;
        LODWORD(v100) = v38;
        v36 = v22[6];
      }
      if ( v36 == 1 )
      {
        v49 = (_DWORD *)*((_QWORD *)v22 + 2);
        v50 = &v49[v22[8]];
        v51 = *v49;
        if ( v49 != v50 )
        {
          do
          {
            v51 = *v49;
            v52 = v49;
            if ( *v49 <= 0xFFFFFFFD )
              goto LABEL_56;
            ++v49;
          }
          while ( v50 != v49 );
          v51 = v52[1];
        }
LABEL_56:
        v53 = *a3;
        v54 = (__int64)&(*a3)[(_QWORD)a3[1]];
        if ( (__int64 *)v54 == *a3 )
          goto LABEL_29;
        v55 = v93;
        v94 = a4;
        v56 = 37 * v51;
        while ( 2 )
        {
          v74 = *v53;
          if ( v55 == *v53 )
            goto LABEL_66;
          v75 = *(unsigned int *)(a1 + 24);
          v76 = *(_QWORD *)(a1 + 8);
          if ( !(_DWORD)v75 )
            goto LABEL_69;
          v57 = (v75 - 1) & (((unsigned int)v74 >> 9) ^ ((unsigned int)v74 >> 4));
          v58 = (__int64 *)(v76 + 16LL * v57);
          v59 = *v58;
          if ( v74 != *v58 )
          {
            v79 = 1;
            while ( v59 != -4096 )
            {
              v57 = (v75 - 1) & (v79 + v57);
              v91 = v79 + 1;
              v58 = (__int64 *)(v76 + 16LL * v57);
              v59 = *v58;
              if ( v74 == *v58 )
                goto LABEL_59;
              v79 = v91;
            }
LABEL_69:
            v58 = (__int64 *)(v76 + 16 * v75);
          }
LABEL_59:
          v60 = *(unsigned int *)(a2 + 24);
          v61 = *((_DWORD *)v58 + 2);
          v62 = *(_QWORD *)(a2 + 8);
          if ( !(_DWORD)v60 )
            goto LABEL_66;
          v63 = (v60 - 1) & (37 * v61);
          v64 = (int *)(v62 + 40LL * v63);
          v65 = *v64;
          if ( v61 != *v64 )
          {
            v78 = 1;
            while ( v65 != -1 )
            {
              v63 = (v60 - 1) & (v78 + v63);
              v90 = v78 + 1;
              v64 = (int *)(v62 + 40LL * v63);
              v65 = *v64;
              if ( v61 == *v64 )
                goto LABEL_61;
              v78 = v90;
            }
            goto LABEL_66;
          }
LABEL_61:
          if ( v64 == (int *)(v62 + 40 * v60) )
            goto LABEL_66;
          v66 = v64[8];
          v67 = *((_QWORD *)v64 + 2);
          if ( v66 )
          {
            v68 = v66 - 1;
            v69 = v68 & v56;
            v70 = (int *)(v67 + 4LL * (v68 & (unsigned int)v56));
            v71 = *v70;
            if ( *v70 == v51 )
            {
LABEL_64:
              *v70 = -2;
              v72 = v64[6];
              ++v64[7];
              v73 = v72 - 1;
              v64[6] = v72 - 1;
LABEL_65:
              if ( !v73 )
                goto LABEL_9;
LABEL_66:
              if ( (__int64 *)v54 == ++v53 )
              {
                a4 = v94;
                goto LABEL_29;
              }
              continue;
            }
            v77 = 1;
            while ( v71 != -1 )
            {
              v83 = v77 + 1;
              v84 = v68 & (unsigned int)(v69 + v77);
              v69 = v84;
              v70 = (int *)(v67 + 4 * v84);
              v71 = *v70;
              if ( *v70 == v51 )
                goto LABEL_64;
              v77 = v83;
            }
          }
          break;
        }
        v73 = v64[6];
        goto LABEL_65;
      }
LABEL_29:
      sub_C7D6A0(v98, 4LL * (unsigned int)v100, 4);
      ++v95;
    }
    while ( (__int64 *)v89 != v95 );
  }
  return 1;
}
