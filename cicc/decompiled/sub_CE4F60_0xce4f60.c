// Function: sub_CE4F60
// Address: 0xce4f60
//
__int64 __fastcall sub_CE4F60(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rbx
  unsigned int v7; // r14d
  __int64 v8; // r8
  __int64 *v9; // rax
  __int64 v10; // r12
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r14
  __int64 v18; // rbx
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  unsigned int v21; // r15d
  __int64 v22; // r12
  __int64 v23; // rdi
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // rsi
  int v27; // eax
  unsigned int v28; // ecx
  __int64 v29; // rdx
  __int64 v30; // r10
  unsigned int v31; // esi
  __int64 v32; // r10
  int v33; // eax
  unsigned int v34; // edi
  __int64 v35; // r10
  unsigned int v36; // esi
  __int64 v37; // r9
  int v38; // r14d
  __int64 *v39; // rcx
  __int64 v40; // r8
  __int64 *v41; // rax
  __int64 v42; // rdi
  __int64 v43; // r14
  unsigned int v44; // eax
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rsi
  _QWORD *v48; // rdx
  __int64 v49; // rcx
  int v50; // edx
  int v51; // ecx
  unsigned __int64 v52; // rcx
  unsigned __int64 v53; // r10
  int v54; // eax
  __int64 v55; // r14
  __int64 result; // rax
  __int64 v57; // r12
  _QWORD *v58; // r14
  _QWORD *i; // r12
  _QWORD *v60; // rsi
  __int64 v61; // r9
  size_t v62; // r15
  int v63; // eax
  int v64; // eax
  int v65; // esi
  int v66; // esi
  unsigned int v67; // edx
  __int64 v68; // rdi
  int v69; // r11d
  __int64 *v70; // r9
  int v71; // esi
  int v72; // esi
  int v73; // r11d
  unsigned int v74; // edx
  __int64 v75; // rdi
  int v76; // r14d
  int v77; // [rsp+8h] [rbp-58h]
  __int64 v78; // [rsp+8h] [rbp-58h]
  unsigned int v79; // [rsp+8h] [rbp-58h]
  __int64 v80; // [rsp+10h] [rbp-50h]
  __int64 v81; // [rsp+18h] [rbp-48h]
  const void *v82; // [rsp+18h] [rbp-48h]
  int s; // [rsp+20h] [rbp-40h]
  void *sa; // [rsp+20h] [rbp-40h]
  __int64 v85[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 96) - *(_QWORD *)(a1 + 88);
  v85[0] = a2;
  v4 = v3 >> 3;
  v5 = sub_22077B0(168);
  v6 = v5;
  if ( v5 )
  {
    *(_QWORD *)v5 = 0;
    v7 = (unsigned int)(v4 + 63) >> 6;
    *(_QWORD *)(v5 + 8) = 0;
    v8 = v5 + 112;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 24) = v5 + 40;
    *(_QWORD *)(v5 + 32) = 0x600000000LL;
    if ( v7 > 6 )
    {
      v82 = (const void *)(v5 + 112);
      sub_C8D5F0(v5 + 24, (const void *)(v5 + 40), v7, 8u, v8, 0x600000000LL);
      memset(*(void **)(v6 + 24), 0, 8LL * v7);
      *(_DWORD *)(v6 + 32) = v7;
      *(_DWORD *)(v6 + 88) = v4;
      *(_QWORD *)(v6 + 96) = v82;
      *(_QWORD *)(v6 + 104) = 0x600000000LL;
      sub_C8D5F0(v6 + 96, v82, v7, 8u, (__int64)v82, 0x600000000LL);
      memset(*(void **)(v6 + 96), 0, 8LL * v7);
      *(_DWORD *)(v6 + 104) = v7;
    }
    else
    {
      if ( v7 )
      {
        v62 = 8LL * v7;
        sa = (void *)(v5 + 112);
        memset((void *)(v5 + 40), 0, v62);
        *(_DWORD *)(v6 + 32) = v7;
        *(_DWORD *)(v6 + 88) = v4;
        *(_QWORD *)(v6 + 96) = sa;
        *(_DWORD *)(v6 + 108) = 6;
        memset(sa, 0, (size_t)sa + v62 - v6 - 112);
      }
      else
      {
        *(_DWORD *)(v5 + 32) = 0;
        *(_DWORD *)(v5 + 88) = v4;
        *(_QWORD *)(v5 + 96) = v8;
        *(_DWORD *)(v5 + 108) = 6;
      }
      *(_DWORD *)(v6 + 104) = v7;
    }
    *(_DWORD *)(v6 + 160) = v4;
  }
  v80 = a1 + 112;
  v9 = sub_CE3FC0(a1 + 112, v85);
  v10 = *v9;
  *v9 = v6;
  if ( v10 )
  {
    v11 = *(_QWORD *)(v10 + 96);
    if ( v11 != v10 + 112 )
      _libc_free(v11, v85);
    v12 = *(_QWORD *)(v10 + 24);
    if ( v12 != v10 + 40 )
      _libc_free(v12, v85);
    j_j___libc_free_0(v10, 168);
  }
  v13 = sub_CE3FC0(v80, v85);
  v17 = v85[0];
  v18 = *v13;
  v19 = v85[0] + 48;
  v20 = *(_QWORD *)(v85[0] + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v20 != v85[0] + 48 )
  {
    if ( !v20 )
      BUG();
    v81 = v20 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v20 - 24) - 30 <= 0xA )
    {
      s = sub_B46E30(v20 - 24);
      if ( s )
      {
        v21 = 0;
        while ( 1 )
        {
          v22 = sub_B46EC0(v81, v21);
          sub_FD0080(a1, v17, v22);
          if ( v85[0] == v22 )
            goto LABEL_40;
          v23 = *(_QWORD *)(a1 + 16);
          if ( v85[0] )
          {
            v24 = (unsigned int)(*(_DWORD *)(v85[0] + 44) + 1);
            v25 = *(_DWORD *)(v85[0] + 44) + 1;
          }
          else
          {
            v24 = 0;
            v25 = 0;
          }
          v15 = *(unsigned int *)(v23 + 32);
          v26 = 0;
          if ( v25 < (unsigned int)v15 )
            v26 = *(_QWORD *)(*(_QWORD *)(v23 + 24) + 8 * v24);
          v27 = *(_DWORD *)(a1 + 168);
          v16 = *(_QWORD *)(a1 + 152);
          if ( v27 )
          {
            v28 = (v27 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
            v29 = v16 + 16LL * v28;
            v30 = *(_QWORD *)v29;
            if ( v26 == *(_QWORD *)v29 )
            {
LABEL_25:
              v31 = *(_DWORD *)(v29 + 8);
              goto LABEL_26;
            }
            v50 = 1;
            while ( v30 != -4096 )
            {
              v76 = v50 + 1;
              v28 = (v27 - 1) & (v50 + v28);
              v29 = v16 + 16LL * v28;
              v30 = *(_QWORD *)v29;
              if ( v26 == *(_QWORD *)v29 )
                goto LABEL_25;
              v50 = v76;
            }
          }
          v31 = 0;
LABEL_26:
          if ( v22 )
          {
            v32 = (unsigned int)(*(_DWORD *)(v22 + 44) + 1);
            v19 = v32;
          }
          else
          {
            v32 = 0;
            v19 = 0;
          }
          v14 = 0;
          if ( (unsigned int)v15 > (unsigned int)v19 )
          {
            v19 = *(_QWORD *)(v23 + 24);
            v14 = *(_QWORD *)(v19 + 8 * v32);
          }
          if ( v27 )
          {
            v33 = v27 - 1;
            v34 = v33 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v19 = v16 + 16LL * v34;
            v35 = *(_QWORD *)v19;
            if ( v14 == *(_QWORD *)v19 )
            {
LABEL_32:
              if ( *(_DWORD *)(v19 + 8) <= v31 )
                goto LABEL_40;
              v36 = *(_DWORD *)(a1 + 136);
              if ( v36 )
              {
                v37 = *(_QWORD *)(a1 + 120);
                v38 = 1;
                v39 = 0;
                v40 = (v36 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
                v41 = (__int64 *)(v37 + 16 * v40);
                v42 = *v41;
                if ( v22 == *v41 )
                {
LABEL_35:
                  v43 = v41[1];
LABEL_36:
                  v44 = *(_DWORD *)(v43 + 88);
                  if ( *(_DWORD *)(v18 + 160) < v44 )
                  {
                    v51 = *(_DWORD *)(v18 + 160) & 0x3F;
                    if ( v51 )
                      *(_QWORD *)(*(_QWORD *)(v18 + 96) + 8LL * *(unsigned int *)(v18 + 104) - 8) &= ~(-1LL << v51);
                    v52 = *(unsigned int *)(v18 + 104);
                    *(_DWORD *)(v18 + 160) = v44;
                    v53 = (v44 + 63) >> 6;
                    if ( v53 != v52 )
                    {
                      if ( v53 >= v52 )
                      {
                        v61 = v53 - v52;
                        if ( v53 > *(unsigned int *)(v18 + 108) )
                        {
                          v78 = v53 - v52;
                          sub_C8D5F0(v18 + 96, (const void *)(v18 + 112), v53, 8u, v40, v61);
                          v52 = *(unsigned int *)(v18 + 104);
                          v61 = v78;
                        }
                        if ( 8 * v61 )
                        {
                          v77 = v61;
                          memset((void *)(*(_QWORD *)(v18 + 96) + 8 * v52), 0, 8 * v61);
                          LODWORD(v52) = *(_DWORD *)(v18 + 104);
                          LODWORD(v61) = v77;
                        }
                        v44 = *(_DWORD *)(v18 + 160);
                        *(_DWORD *)(v18 + 104) = v61 + v52;
                      }
                      else
                      {
                        *(_DWORD *)(v18 + 104) = (v44 + 63) >> 6;
                      }
                    }
                    v54 = v44 & 0x3F;
                    if ( v54 )
                      *(_QWORD *)(*(_QWORD *)(v18 + 96) + 8LL * *(unsigned int *)(v18 + 104) - 8) &= ~(-1LL << v54);
                  }
                  v45 = 0;
                  v46 = *(unsigned int *)(v43 + 32);
                  v47 = 8 * v46;
                  if ( (_DWORD)v46 )
                  {
                    do
                    {
                      v48 = (_QWORD *)(v45 + *(_QWORD *)(v18 + 96));
                      v49 = *(_QWORD *)(*(_QWORD *)(v43 + 24) + v45);
                      v45 += 8;
                      *v48 |= v49;
                    }
                    while ( v45 != v47 );
                  }
                  sub_FCEE70(a1, v85[0], v22);
                  goto LABEL_40;
                }
                while ( v42 != -4096 )
                {
                  if ( v42 == -8192 && !v39 )
                    v39 = v41;
                  v40 = (v36 - 1) & (v38 + (_DWORD)v40);
                  v41 = (__int64 *)(v37 + 16LL * (unsigned int)v40);
                  v42 = *v41;
                  if ( v22 == *v41 )
                    goto LABEL_35;
                  ++v38;
                }
                if ( !v39 )
                  v39 = v41;
                v63 = *(_DWORD *)(a1 + 128);
                ++*(_QWORD *)(a1 + 112);
                v64 = v63 + 1;
                if ( 4 * v64 < 3 * v36 )
                {
                  v40 = v36 >> 3;
                  if ( v36 - *(_DWORD *)(a1 + 132) - v64 <= (unsigned int)v40 )
                  {
                    v79 = ((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4);
                    sub_CE3D80(v80, v36);
                    v71 = *(_DWORD *)(a1 + 136);
                    if ( !v71 )
                    {
LABEL_107:
                      ++*(_DWORD *)(a1 + 128);
                      BUG();
                    }
                    v72 = v71 - 1;
                    v40 = *(_QWORD *)(a1 + 120);
                    v73 = 1;
                    v70 = 0;
                    v74 = v72 & v79;
                    v64 = *(_DWORD *)(a1 + 128) + 1;
                    v39 = (__int64 *)(v40 + 16LL * (v72 & v79));
                    v75 = *v39;
                    if ( v22 != *v39 )
                    {
                      while ( v75 != -4096 )
                      {
                        if ( v75 == -8192 && !v70 )
                          v70 = v39;
                        v74 = v72 & (v74 + v73);
                        v39 = (__int64 *)(v40 + 16LL * v74);
                        v75 = *v39;
                        if ( v22 == *v39 )
                          goto LABEL_75;
                        ++v73;
                      }
                      goto LABEL_88;
                    }
                  }
                  goto LABEL_75;
                }
              }
              else
              {
                ++*(_QWORD *)(a1 + 112);
              }
              sub_CE3D80(v80, 2 * v36);
              v65 = *(_DWORD *)(a1 + 136);
              if ( !v65 )
                goto LABEL_107;
              v66 = v65 - 1;
              v40 = *(_QWORD *)(a1 + 120);
              v67 = v66 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
              v64 = *(_DWORD *)(a1 + 128) + 1;
              v39 = (__int64 *)(v40 + 16LL * v67);
              v68 = *v39;
              if ( v22 != *v39 )
              {
                v69 = 1;
                v70 = 0;
                while ( v68 != -4096 )
                {
                  if ( v68 == -8192 && !v70 )
                    v70 = v39;
                  v67 = v66 & (v67 + v69);
                  v39 = (__int64 *)(v40 + 16LL * v67);
                  v68 = *v39;
                  if ( v22 == *v39 )
                    goto LABEL_75;
                  ++v69;
                }
LABEL_88:
                if ( v70 )
                  v39 = v70;
              }
LABEL_75:
              *(_DWORD *)(a1 + 128) = v64;
              if ( *v39 != -4096 )
                --*(_DWORD *)(a1 + 132);
              *v39 = v22;
              v43 = 0;
              v39[1] = 0;
              goto LABEL_36;
            }
            v19 = 1;
            while ( v35 != -4096 )
            {
              v15 = (unsigned int)(v19 + 1);
              v34 = v33 & (v19 + v34);
              v19 = v16 + 16LL * v34;
              v35 = *(_QWORD *)v19;
              if ( v14 == *(_QWORD *)v19 )
                goto LABEL_32;
              v19 = (unsigned int)v15;
            }
          }
LABEL_40:
          if ( s == ++v21 )
            break;
          v17 = v85[0];
        }
      }
    }
  }
  sub_CE14D0(v18 + 24, v18 + 96, v19, v14, v15, v16);
  v55 = v85[0];
  result = *(unsigned int *)(v18 + 160);
  *(_DWORD *)(v18 + 88) = result;
  v57 = *(_QWORD *)(v55 + 48);
  v58 = (_QWORD *)(v55 + 48);
  for ( i = (_QWORD *)(v57 & 0xFFFFFFFFFFFFFFF8LL); v58 != i; i = (_QWORD *)(*i & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v60 = i - 3;
    if ( !i )
      v60 = 0;
    result = sub_FD0480(a1, v60, v18);
  }
  return result;
}
