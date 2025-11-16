// Function: sub_2512BA0
// Address: 0x2512ba0
//
__int64 __fastcall sub_2512BA0(__int64 a1, _QWORD *a2, int a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // rbx
  unsigned __int64 v8; // r13
  __int64 v9; // r15
  int v10; // esi
  unsigned __int64 v11; // rax
  unsigned int i; // eax
  __int64 v13; // rcx
  unsigned int v14; // eax
  __int64 v16; // rax
  int v17; // r15d
  unsigned __int64 v18; // rax
  unsigned int v19; // esi
  __int64 v20; // rdx
  __int64 v21; // r8
  int v22; // r15d
  __int64 *v23; // r11
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // r13
  unsigned __int64 *v28; // r15
  __int64 v29; // rax
  __int64 v30; // rax
  size_t v31; // r15
  char *v32; // rax
  char *v33; // r12
  __int64 v34; // r15
  char *v35; // rbx
  unsigned __int64 v36; // r8
  unsigned __int64 v37; // rsi
  int v38; // ecx
  unsigned int v39; // eax
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // r9
  __int64 v43; // rdx
  char *v44; // rax
  int v45; // r10d
  int v46; // r9d
  unsigned int v47; // eax
  __int64 v48; // rsi
  __int64 v49; // rax
  int v50; // ecx
  int v51; // ecx
  __int64 v52; // rax
  unsigned __int64 v53; // r8
  __int64 v54; // [rsp+8h] [rbp-D8h]
  __int64 n; // [rsp+10h] [rbp-D0h]
  __int64 *v56; // [rsp+18h] [rbp-C8h]
  __int64 v57; // [rsp+20h] [rbp-C0h]
  char *v58; // [rsp+28h] [rbp-B8h]
  void *v59; // [rsp+30h] [rbp-B0h]
  __int64 v60; // [rsp+38h] [rbp-A8h]
  int v61; // [rsp+40h] [rbp-A0h]
  __int64 v63; // [rsp+48h] [rbp-98h]
  int v65; // [rsp+58h] [rbp-88h]
  __int64 v66; // [rsp+58h] [rbp-88h]
  __int64 v67; // [rsp+58h] [rbp-88h]
  unsigned __int64 v68; // [rsp+58h] [rbp-88h]
  unsigned __int64 v69; // [rsp+68h] [rbp-78h] BYREF
  __int64 *v70; // [rsp+70h] [rbp-70h] BYREF
  void *v71; // [rsp+78h] [rbp-68h]
  __int64 v72; // [rsp+80h] [rbp-60h]
  __int64 v73; // [rsp+88h] [rbp-58h]
  __int64 v74; // [rsp+90h] [rbp-50h]
  __int64 v75; // [rsp+98h] [rbp-48h]
  __int64 v76; // [rsp+A0h] [rbp-40h]
  __int64 v77; // [rsp+A8h] [rbp-38h]

  LODWORD(v4) = 0;
  v5 = *(_QWORD *)(a1 + 208);
  v6 = *(_QWORD *)(v5 + 120);
  if ( v6 )
  {
    v8 = sub_250D070(a2);
    v65 = *(_DWORD *)(v5 + 152);
    if ( !v65 )
      goto LABEL_8;
    v9 = *(_QWORD *)(v5 + 136);
    LODWORD(v70) = a3;
    v10 = 1;
    v11 = 0xBF58476D1CE4E5B9LL
        * ((unsigned int)sub_CF97C0((unsigned int *)&v70)
         | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32));
    for ( i = (v65 - 1) & ((v11 >> 31) ^ v11); ; i = (v65 - 1) & v14 )
    {
      v13 = v9 + 48LL * i;
      if ( v8 == *(_QWORD *)v13 && a3 == *(_DWORD *)(v13 + 8) )
        break;
      if ( *(_QWORD *)v13 == -4096 && *(_DWORD *)(v13 + 8) == 100 )
        goto LABEL_8;
      v14 = v10 + i;
      ++v10;
    }
    v66 = v9 + 48LL * i;
    sub_C7D6A0(0, 0, 8);
    v16 = *(unsigned int *)(v66 + 40);
    if ( (_DWORD)v16 )
    {
      v57 = 24 * v16;
      n = 24 * v16;
      v58 = (char *)sub_C7D670(24 * v16, 8);
      v17 = *(_DWORD *)(v66 + 32);
      memcpy(v58, *(const void **)(v66 + 24), n);
      if ( v17 )
      {
        v56 = (__int64 *)sub_BD5C60(v8);
        v61 = *(_DWORD *)(a4 + 8);
        v18 = sub_2509740(a2);
        v19 = *(_DWORD *)(v6 + 192);
        v69 = v18;
        v20 = v18;
        if ( v19 )
        {
          v21 = *(_QWORD *)(v6 + 176);
          v22 = 1;
          v23 = 0;
          v24 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v25 = (__int64 *)(v21 + 16LL * v24);
          v26 = *v25;
          if ( v20 == *v25 )
          {
LABEL_16:
            v27 = v25[1];
            v28 = (unsigned __int64 *)(v25 + 1);
            if ( v27 )
              goto LABEL_17;
            goto LABEL_64;
          }
          while ( v26 != -4096 )
          {
            if ( v26 == -8192 && !v23 )
              v23 = v25;
            v24 = (v19 - 1) & (v22 + v24);
            v25 = (__int64 *)(v21 + 16LL * v24);
            v26 = *v25;
            if ( v20 == *v25 )
              goto LABEL_16;
            ++v22;
          }
          if ( v23 )
            v25 = v23;
          v70 = v25;
          v50 = *(_DWORD *)(v6 + 184);
          ++*(_QWORD *)(v6 + 168);
          v51 = v50 + 1;
          if ( 4 * v51 < 3 * v19 )
          {
            if ( v19 - *(_DWORD *)(v6 + 188) - v51 > v19 >> 3 )
            {
LABEL_61:
              *(_DWORD *)(v6 + 184) = v51;
              if ( *v25 != -4096 )
                --*(_DWORD *)(v6 + 188);
              *v25 = v20;
              v28 = (unsigned __int64 *)(v25 + 1);
              v25[1] = 0;
LABEL_64:
              v52 = sub_22077B0(0x40u);
              v27 = v52;
              if ( v52 )
                sub_3106C40(v52, v6, v69);
              v53 = *v28;
              *v28 = v27;
              if ( v53 )
              {
                v68 = v53;
                sub_C7D6A0(*(_QWORD *)(v53 + 8), 8LL * *(unsigned int *)(v53 + 24), 8);
                j_j___libc_free_0(v68);
                v27 = *v28;
              }
LABEL_17:
              v70 = 0;
              v71 = 0;
              v72 = 0;
              v73 = 0;
              sub_C7D6A0(0, 0, 8);
              v29 = *(unsigned int *)(v27 + 24);
              LODWORD(v73) = v29;
              if ( (_DWORD)v29 )
              {
                v71 = (void *)sub_C7D670(8 * v29, 8);
                v72 = *(_QWORD *)(v27 + 16);
                memcpy(v71, *(const void **)(v27 + 8), 8LL * (unsigned int)v73);
              }
              else
              {
                v71 = 0;
                v72 = 0;
              }
              v74 = *(_QWORD *)(v27 + 32);
              v75 = *(_QWORD *)(v27 + 40);
              v76 = *(_QWORD *)(v27 + 48);
              v77 = *(_QWORD *)(v27 + 56);
              sub_2509740(a2);
              sub_C7D6A0(0, 0, 8);
              v30 = *(unsigned int *)(v6 + 224);
              if ( (_DWORD)v30 )
              {
                v31 = 8 * v30;
                v60 = 8 * v30;
                v59 = (void *)sub_C7D670(8 * v30, 8);
                memcpy(v59, *(const void **)(v6 + 208), v31);
              }
              else
              {
                v60 = 0;
                v59 = 0;
              }
              v4 = *(_QWORD *)(v6 + 240);
              v67 = *(_QWORD *)(v6 + 248);
              v63 = *(_QWORD *)(v6 + 256);
              v32 = v58;
              v33 = &v58[n];
              while ( 1 )
              {
                v34 = *(_QWORD *)v32;
                v35 = v32;
                if ( *(_QWORD *)v32 != -4096 && v34 != -8192 )
                  break;
                v32 += 24;
                if ( v33 == v32 )
                  goto LABEL_25;
              }
              if ( v33 != v32 )
              {
                while ( 1 )
                {
                  v36 = v34 & 0xFFFFFFFFFFFFFFFBLL;
                  v37 = v34 | 4;
                  if ( !(_DWORD)v73 )
                    goto LABEL_41;
                  v38 = v73 - 1;
                  v39 = (v73 - 1) & (v37 ^ (v37 >> 9));
                  v40 = *((_QWORD *)v71 + v39);
                  if ( v37 != v40 )
                  {
                    v45 = 1;
                    while ( v40 != -4 )
                    {
                      v39 = v38 & (v45 + v39);
                      v40 = *((_QWORD *)v71 + v39);
                      if ( v37 == v40 )
                        goto LABEL_29;
                      ++v45;
                    }
                    v46 = 1;
                    v47 = v38 & (v36 ^ (v36 >> 9));
                    v48 = *((_QWORD *)v71 + v47);
                    if ( v36 != v48 )
                      break;
                  }
LABEL_29:
                  v41 = sub_A778C0(v56, a3, *((_QWORD *)v35 + 2));
                  v43 = *(unsigned int *)(a4 + 8);
                  if ( v43 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
                  {
                    v54 = v41;
                    sub_C8D5F0(a4, (const void *)(a4 + 16), v43 + 1, 8u, v43 + 1, v42);
                    v43 = *(unsigned int *)(a4 + 8);
                    v41 = v54;
                  }
                  *(_QWORD *)(*(_QWORD *)a4 + 8 * v43) = v41;
                  ++*(_DWORD *)(a4 + 8);
LABEL_32:
                  v44 = v35 + 24;
                  if ( v35 + 24 != v33 )
                  {
                    while ( 1 )
                    {
                      v34 = *(_QWORD *)v44;
                      v35 = v44;
                      if ( *(_QWORD *)v44 != -4096 && v34 != -8192 )
                        break;
                      v44 += 24;
                      if ( v33 == v44 )
                        goto LABEL_25;
                    }
                    if ( v33 != v44 )
                      continue;
                  }
                  goto LABEL_25;
                }
                while ( v48 != -4 )
                {
                  v47 = v38 & (v46 + v47);
                  v48 = *((_QWORD *)v71 + v47);
                  if ( v36 == v48 )
                    goto LABEL_29;
                  ++v46;
                }
LABEL_41:
                v49 = v75;
                while ( v4 != v49 || v67 != v76 || v63 != v77 )
                {
                  v49 = sub_3106C80(&v70);
                  v75 = v49;
                  if ( v49 == v34 )
                    goto LABEL_29;
                }
                goto LABEL_32;
              }
LABEL_25:
              LOBYTE(v4) = *(_DWORD *)(a4 + 8) != v61;
              sub_C7D6A0((__int64)v59, v60, 8);
              sub_C7D6A0((__int64)v71, 8LL * (unsigned int)v73, 8);
              goto LABEL_9;
            }
LABEL_74:
            sub_2512960(v6 + 168, v19);
            sub_2510430(v6 + 168, (__int64 *)&v69, &v70);
            v20 = v69;
            v51 = *(_DWORD *)(v6 + 184) + 1;
            v25 = v70;
            goto LABEL_61;
          }
        }
        else
        {
          v70 = 0;
          ++*(_QWORD *)(v6 + 168);
        }
        v19 *= 2;
        goto LABEL_74;
      }
      LODWORD(v4) = 0;
    }
    else
    {
LABEL_8:
      v57 = 0;
      LODWORD(v4) = 0;
      v58 = 0;
    }
LABEL_9:
    sub_C7D6A0((__int64)v58, v57, 8);
  }
  return (unsigned int)v4;
}
