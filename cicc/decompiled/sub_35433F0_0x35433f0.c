// Function: sub_35433F0
// Address: 0x35433f0
//
__int64 __fastcall sub_35433F0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 *v7; // rbx
  __int64 v8; // r15
  __int64 *v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 *v12; // rdi
  int v13; // r9d
  __int64 v14; // rsi
  int v15; // ecx
  __int64 v16; // rdx
  unsigned int v17; // r10d
  __int64 v18; // r8
  int v19; // ecx
  int v20; // ecx
  __int64 v21; // r8
  unsigned int v22; // eax
  __int64 *v23; // rsi
  __int64 v24; // rdi
  __int64 *v25; // rcx
  int v26; // eax
  unsigned int v27; // r9d
  __int64 v28; // r8
  int v29; // eax
  int v30; // eax
  __int64 v31; // r9
  unsigned int v32; // esi
  __int64 *v33; // rdi
  __int64 v34; // r8
  __int64 v35; // rdx
  __int64 v36; // rdi
  int v37; // eax
  __int64 v38; // rax
  __int64 *v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // rsi
  unsigned int v43; // eax
  __int64 v44; // rbx
  __int64 v45; // r13
  char v46; // al
  __int64 v47; // rax
  unsigned __int64 v48; // rdi
  int v49; // eax
  int v50; // eax
  int v51; // r14d
  int v52; // r11d
  __int64 v53; // rdx
  __int64 *v54; // r11
  unsigned int v55; // eax
  unsigned int v56; // r10d
  __int64 v57; // r8
  int v58; // ecx
  int v59; // ecx
  __int64 v60; // r8
  unsigned int v61; // eax
  __int64 *v62; // rsi
  __int64 v63; // rdi
  int v64; // esi
  int v65; // r14d
  __int64 v66; // rdx
  unsigned int v67; // eax
  unsigned int v68; // r10d
  __int64 v69; // r8
  int v70; // ecx
  int v71; // ecx
  __int64 v72; // r8
  unsigned int v73; // eax
  __int64 v74; // rdi
  int v75; // esi
  int v76; // r9d
  int v77; // r14d
  __int64 v78; // rdx
  unsigned int v79; // eax
  unsigned int v80; // r10d
  __int64 v81; // r8
  int v82; // ecx
  int v83; // ecx
  __int64 v84; // r8
  unsigned int v85; // eax
  __int64 v86; // rdi
  int v87; // esi
  int v88; // r9d
  int v89; // r10d
  __int64 v90; // rax
  __int64 v91; // rbx
  unsigned __int64 v92; // rdi
  int v93; // esi
  int v94; // r9d
  int v95; // r9d
  int v96; // edi
  int v97; // r10d
  __int64 v99; // [rsp+28h] [rbp-58h]
  __int64 v100; // [rsp+38h] [rbp-48h] BYREF
  __int64 *v101[8]; // [rsp+40h] [rbp-40h] BYREF

  v2 = *((unsigned int *)a2 + 2);
  v3 = *a2;
  v100 = v3;
  result = v3 + 88 * v2;
  v99 = result;
  if ( result != v3 )
  {
    while ( 1 )
    {
      v5 = v3 + 88;
      if ( v5 != v99 )
        break;
LABEL_41:
      result = v100;
      v3 = v100 + 88;
      v100 = v3;
      if ( v3 == v99 )
        return result;
    }
    while ( 1 )
    {
      v6 = *(unsigned int *)(v5 + 40);
      v101[1] = (__int64 *)v5;
      v7 = *(__int64 **)(v5 + 32);
      LODWORD(v8) = v6;
      v6 *= 8;
      v101[0] = &v100;
      v9 = (__int64 *)((char *)v7 + v6);
      v10 = v6 >> 3;
      v11 = v6 >> 5;
      if ( v11 )
      {
        v12 = &v7[4 * v11];
        v13 = *(_DWORD *)(v100 + 24);
        v14 = *(_QWORD *)(v100 + 8);
        v15 = v13 - 1;
        do
        {
          v16 = *v7;
          if ( v13 )
          {
            v17 = v15 & (((unsigned int)v16 >> 4) ^ ((unsigned int)v16 >> 9));
            v18 = *(_QWORD *)(v14 + 8LL * v17);
            if ( v16 == v18 )
            {
LABEL_7:
              v19 = *(_DWORD *)(v5 + 24);
              if ( v19 )
              {
                v20 = v19 - 1;
                v21 = *(_QWORD *)(v5 + 8);
                v22 = v20 & (((unsigned int)v16 >> 4) ^ ((unsigned int)v16 >> 9));
                v23 = (__int64 *)(v21 + 8LL * v22);
                v24 = *v23;
                if ( v16 == *v23 )
                {
LABEL_9:
                  *v23 = -8192;
                  --*(_DWORD *)(v5 + 16);
                  ++*(_DWORD *)(v5 + 20);
                }
                else
                {
                  v93 = 1;
                  while ( v24 != -4096 )
                  {
                    v94 = v93 + 1;
                    v22 = v20 & (v93 + v22);
                    v23 = (__int64 *)(v21 + 8LL * v22);
                    v24 = *v23;
                    if ( v16 == *v23 )
                      goto LABEL_9;
                    v93 = v94;
                  }
                }
              }
              goto LABEL_10;
            }
            v52 = 1;
            while ( v18 != -4096 )
            {
              v17 = v15 & (v52 + v17);
              v18 = *(_QWORD *)(v14 + 8LL * v17);
              if ( v16 == v18 )
                goto LABEL_7;
              ++v52;
            }
            v53 = v7[1];
            v54 = v7 + 1;
            v55 = ((unsigned int)v53 >> 4) ^ ((unsigned int)v53 >> 9);
            v56 = v55 & v15;
            v57 = *(_QWORD *)(v14 + 8LL * (v55 & v15));
            if ( v53 == v57 )
            {
LABEL_46:
              v58 = *(_DWORD *)(v5 + 24);
              if ( v58 )
              {
                v59 = v58 - 1;
                v60 = *(_QWORD *)(v5 + 8);
                v61 = v59 & v55;
                v62 = (__int64 *)(v60 + 8LL * v61);
                v63 = *v62;
                if ( v53 == *v62 )
                  goto LABEL_48;
                v64 = 1;
                while ( v63 != -4096 )
                {
                  v95 = v64 + 1;
                  v61 = v59 & (v64 + v61);
                  v62 = (__int64 *)(v60 + 8LL * v61);
                  v63 = *v62;
                  if ( v53 == *v62 )
                    goto LABEL_48;
                  v64 = v95;
                }
              }
LABEL_51:
              v7 = v54;
LABEL_10:
              if ( v9 == v7 || (v25 = v7 + 1, v9 == v7 + 1) )
              {
LABEL_20:
                v38 = *(_QWORD *)(v5 + 32);
                v8 = *(unsigned int *)(v5 + 40);
                v39 = (__int64 *)(v38 + 8 * v8);
                goto LABEL_21;
              }
              while ( 1 )
              {
LABEL_18:
                v35 = *v25;
                v36 = *(_QWORD *)(v100 + 8);
                v37 = *(_DWORD *)(v100 + 24);
                if ( !v37 )
                  goto LABEL_19;
                v26 = v37 - 1;
                v27 = v26 & (((unsigned int)v35 >> 4) ^ ((unsigned int)v35 >> 9));
                v28 = *(_QWORD *)(v36 + 8LL * v27);
                if ( v28 != v35 )
                  break;
LABEL_14:
                v29 = *(_DWORD *)(v5 + 24);
                if ( v29 )
                {
                  v30 = v29 - 1;
                  v31 = *(_QWORD *)(v5 + 8);
                  v32 = v30 & (((unsigned int)v35 >> 4) ^ ((unsigned int)v35 >> 9));
                  v33 = (__int64 *)(v31 + 8LL * v32);
                  v34 = *v33;
                  if ( *v33 == v35 )
                  {
LABEL_16:
                    *v33 = -8192;
                    --*(_DWORD *)(v5 + 16);
                    ++*(_DWORD *)(v5 + 20);
                  }
                  else
                  {
                    v96 = 1;
                    while ( v34 != -4096 )
                    {
                      v97 = v96 + 1;
                      v32 = v30 & (v96 + v32);
                      v33 = (__int64 *)(v31 + 8LL * v32);
                      v34 = *v33;
                      if ( v35 == *v33 )
                        goto LABEL_16;
                      v96 = v97;
                    }
                  }
                }
                if ( v9 == ++v25 )
                  goto LABEL_20;
              }
              v89 = 1;
              while ( v28 != -4096 )
              {
                v27 = v26 & (v89 + v27);
                v28 = *(_QWORD *)(v36 + 8LL * v27);
                if ( v35 == v28 )
                  goto LABEL_14;
                ++v89;
              }
LABEL_19:
              ++v25;
              *v7++ = v35;
              if ( v9 == v25 )
                goto LABEL_20;
              goto LABEL_18;
            }
            v65 = 1;
            while ( v57 != -4096 )
            {
              v56 = v15 & (v65 + v56);
              v57 = *(_QWORD *)(v14 + 8LL * v56);
              if ( v53 == v57 )
                goto LABEL_46;
              ++v65;
            }
            v66 = v7[2];
            v54 = v7 + 2;
            v67 = ((unsigned int)v66 >> 4) ^ ((unsigned int)v66 >> 9);
            v68 = v67 & v15;
            v69 = *(_QWORD *)(v14 + 8LL * (v67 & v15));
            if ( v69 == v66 )
            {
LABEL_55:
              v70 = *(_DWORD *)(v5 + 24);
              if ( !v70 )
                goto LABEL_51;
              v71 = v70 - 1;
              v72 = *(_QWORD *)(v5 + 8);
              v73 = v71 & v67;
              v62 = (__int64 *)(v72 + 8LL * v73);
              v74 = *v62;
              if ( v66 != *v62 )
              {
                v75 = 1;
                while ( v74 != -4096 )
                {
                  v76 = v75 + 1;
                  v73 = v71 & (v75 + v73);
                  v62 = (__int64 *)(v72 + 8LL * v73);
                  v74 = *v62;
                  if ( v66 == *v62 )
                    goto LABEL_48;
                  v75 = v76;
                }
                goto LABEL_51;
              }
LABEL_48:
              *v62 = -8192;
              v7 = v54;
              --*(_DWORD *)(v5 + 16);
              ++*(_DWORD *)(v5 + 20);
              goto LABEL_10;
            }
            v77 = 1;
            while ( v69 != -4096 )
            {
              v68 = v15 & (v77 + v68);
              v69 = *(_QWORD *)(v14 + 8LL * v68);
              if ( v66 == v69 )
                goto LABEL_55;
              ++v77;
            }
            v78 = v7[3];
            v54 = v7 + 3;
            v79 = ((unsigned int)v78 >> 4) ^ ((unsigned int)v78 >> 9);
            v80 = v79 & v15;
            v81 = *(_QWORD *)(v14 + 8LL * (v79 & v15));
            if ( v81 == v78 )
            {
LABEL_64:
              v82 = *(_DWORD *)(v5 + 24);
              if ( !v82 )
                goto LABEL_51;
              v83 = v82 - 1;
              v84 = *(_QWORD *)(v5 + 8);
              v85 = v83 & v79;
              v62 = (__int64 *)(v84 + 8LL * v85);
              v86 = *v62;
              if ( *v62 != v78 )
              {
                v87 = 1;
                while ( v86 != -4096 )
                {
                  v88 = v87 + 1;
                  v85 = v83 & (v87 + v85);
                  v62 = (__int64 *)(v84 + 8LL * v85);
                  v86 = *v62;
                  if ( v78 == *v62 )
                    goto LABEL_48;
                  v87 = v88;
                }
                goto LABEL_51;
              }
              goto LABEL_48;
            }
            v51 = 1;
            while ( v81 != -4096 )
            {
              v80 = v15 & (v51 + v80);
              v81 = *(_QWORD *)(v14 + 8LL * v80);
              if ( v78 == v81 )
                goto LABEL_64;
              ++v51;
            }
          }
          v7 += 4;
        }
        while ( v12 != v7 );
        v10 = v9 - v7;
      }
      if ( v10 != 2 )
      {
        if ( v10 != 3 )
        {
          if ( v10 != 1 )
            goto LABEL_38;
          goto LABEL_94;
        }
        if ( (unsigned __int8)sub_353E8A0(v101, v7) )
          goto LABEL_10;
        ++v7;
      }
      if ( (unsigned __int8)sub_353E8A0(v101, v7) )
        goto LABEL_10;
      ++v7;
LABEL_94:
      if ( (unsigned __int8)sub_353E8A0(v101, v7) )
        goto LABEL_10;
      v38 = *(_QWORD *)(v5 + 32);
      v7 = v9;
      v8 = *(unsigned int *)(v5 + 40);
      v39 = (__int64 *)(v38 + 8 * v8);
LABEL_21:
      if ( v7 != v39 )
      {
        v40 = v5 + 88;
        v41 = ((__int64)v7 - v38) >> 3;
        *(_DWORD *)(v5 + 40) = v41;
        if ( !(_DWORD)v41 )
          goto LABEL_23;
        goto LABEL_39;
      }
LABEL_38:
      v40 = v5 + 88;
      if ( !(_DWORD)v8 )
      {
LABEL_23:
        v42 = *a2;
        v43 = *((_DWORD *)a2 + 2);
        v44 = 0x2E8BA2E8BA2E8BA3LL * ((*a2 + 88LL * v43 - v40) >> 3);
        if ( *a2 + 88LL * v43 - v40 > 0 )
        {
          v45 = v5 + 120;
          do
          {
            sub_C7D6A0(*(_QWORD *)(v45 - 112), 8LL * *(unsigned int *)(v45 - 96), 8);
            v47 = *(_QWORD *)(v45 - 24);
            ++*(_QWORD *)(v45 - 120);
            ++*(_QWORD *)(v45 - 32);
            *(_QWORD *)(v45 - 112) = v47;
            LODWORD(v47) = *(_DWORD *)(v45 - 16);
            *(_QWORD *)(v45 - 24) = 0;
            *(_DWORD *)(v45 - 104) = v47;
            LODWORD(v47) = *(_DWORD *)(v45 - 12);
            *(_DWORD *)(v45 - 16) = 0;
            *(_DWORD *)(v45 - 100) = v47;
            LODWORD(v47) = *(_DWORD *)(v45 - 8);
            *(_DWORD *)(v45 - 12) = 0;
            *(_DWORD *)(v45 - 96) = v47;
            LODWORD(v47) = *(_DWORD *)(v45 + 8);
            *(_DWORD *)(v45 - 8) = 0;
            if ( (_DWORD)v47 )
            {
              v48 = *(_QWORD *)(v45 - 88);
              if ( v48 != v45 - 72 )
                _libc_free(v48);
              *(_QWORD *)(v45 - 88) = *(_QWORD *)v45;
              v49 = *(_DWORD *)(v45 + 8);
              *(_DWORD *)(v45 + 8) = 0;
              *(_DWORD *)(v45 - 80) = v49;
              v50 = *(_DWORD *)(v45 + 12);
              *(_DWORD *)(v45 + 12) = 0;
              *(_DWORD *)(v45 - 76) = v50;
              *(_QWORD *)v45 = v45 + 16;
            }
            else
            {
              *(_DWORD *)(v45 - 80) = 0;
            }
            v46 = *(_BYTE *)(v45 + 16);
            v45 += 88;
            *(_BYTE *)(v45 - 160) = v46;
            *(_DWORD *)(v45 - 156) = *(_DWORD *)(v45 - 68);
            *(_DWORD *)(v45 - 152) = *(_DWORD *)(v45 - 64);
            *(_DWORD *)(v45 - 148) = *(_DWORD *)(v45 - 60);
            *(_DWORD *)(v45 - 144) = *(_DWORD *)(v45 - 56);
            *(_QWORD *)(v45 - 136) = *(_QWORD *)(v45 - 48);
            *(_DWORD *)(v45 - 128) = *(_DWORD *)(v45 - 40);
            --v44;
          }
          while ( v44 );
          v43 = *((_DWORD *)a2 + 2);
          v42 = *a2;
        }
        v90 = v43 - 1;
        *((_DWORD *)a2 + 2) = v90;
        v91 = v42 + 88 * v90;
        v92 = *(_QWORD *)(v91 + 32);
        if ( v92 != v91 + 48 )
          _libc_free(v92);
        sub_C7D6A0(*(_QWORD *)(v91 + 8), 8LL * *(unsigned int *)(v91 + 24), 8);
        v99 = *a2 + 88LL * *((unsigned int *)a2 + 2);
        goto LABEL_40;
      }
LABEL_39:
      v5 = v40;
LABEL_40:
      if ( v5 == v99 )
        goto LABEL_41;
    }
  }
  return result;
}
