// Function: sub_1F13A50
// Address: 0x1f13a50
//
__int64 __fastcall sub_1F13A50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 *v9; // r10
  __int64 v10; // rax
  __int64 *v11; // rbx
  __int64 *v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r12
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rax
  __int64 j; // rcx
  __int64 v21; // rdi
  __int64 v22; // rcx
  unsigned int v23; // esi
  __int64 *v24; // rdx
  __int64 v25; // r10
  __int64 v26; // r15
  __int16 v27; // ax
  __int64 v28; // rdx
  unsigned __int64 v29; // rdx
  __int64 v30; // rax
  __int64 k; // rdx
  __int64 v32; // rsi
  unsigned int v33; // edx
  unsigned int v34; // ecx
  __int64 *v35; // rax
  __int64 v36; // r11
  _QWORD *v37; // r15
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 v40; // rbx
  __int64 v41; // rbx
  __int64 *v42; // rax
  unsigned __int64 v43; // rax
  __int64 v44; // r15
  __int64 v45; // rbx
  __int64 *v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned int v49; // ecx
  __int64 v50; // rbx
  __int64 *v51; // rax
  __int64 v52; // rbx
  __int64 *v53; // rax
  __int64 v54; // rbx
  __int64 *v55; // rax
  __int64 v56; // rbx
  __int64 *v57; // rax
  __int64 v58; // rbx
  __int64 *v59; // rax
  __int64 v60; // rbx
  __int64 *v61; // rax
  int v62; // eax
  int v63; // edx
  int v64; // r8d
  int v65; // r8d
  __int64 *v66; // [rsp+8h] [rbp-78h]
  _QWORD *v67; // [rsp+10h] [rbp-70h]
  __int64 v68; // [rsp+10h] [rbp-70h]
  __int64 v69; // [rsp+18h] [rbp-68h]
  _QWORD *v70; // [rsp+28h] [rbp-58h]
  _BYTE *v71; // [rsp+30h] [rbp-50h] BYREF
  __int64 i; // [rsp+38h] [rbp-48h]
  _BYTE v73[64]; // [rsp+40h] [rbp-40h] BYREF

  v9 = *(__int64 **)(a3 + 96);
  v10 = 16LL * *(unsigned int *)(a3 + 48);
  v11 = *(__int64 **)(a3 + 88);
  v12 = (__int64 *)(v10 + a1[1]);
  v69 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 272LL) + 392LL) + v10 + 8);
  v71 = v73;
  for ( i = 0x100000000LL; v9 != v11; LODWORD(i) = i + 1 )
  {
    while ( 1 )
    {
      v13 = *v11;
      if ( *(_BYTE *)(*v11 + 180) )
        break;
      if ( v9 == ++v11 )
        goto LABEL_8;
    }
    v14 = (unsigned int)i;
    if ( (unsigned int)i >= HIDWORD(i) )
    {
      v66 = v9;
      v68 = *v11;
      sub_16CD150((__int64)&v71, v73, 0, 8, a5, a6);
      v14 = (unsigned int)i;
      v9 = v66;
      v13 = v68;
    }
    ++v11;
    *(_QWORD *)&v71[8 * v14] = v13;
  }
LABEL_8:
  if ( (*v12 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    v18 = a3 + 24;
    v19 = sub_1DD5EE0(a3);
    if ( a3 + 24 == v19 )
    {
      *v12 = v69;
      goto LABEL_21;
    }
    for ( j = *(_QWORD *)(*a1 + 272LL); (*(_BYTE *)(v19 + 46) & 4) != 0; v19 = *(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF8LL )
      ;
    v21 = *(_QWORD *)(j + 368);
    v22 = *(unsigned int *)(j + 384);
    if ( (_DWORD)v22 )
    {
      v23 = (v22 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v24 = (__int64 *)(v21 + 16LL * v23);
      v25 = *v24;
      if ( *v24 == v19 )
      {
LABEL_20:
        *v12 = v24[1];
LABEL_21:
        if ( !(_DWORD)i )
          goto LABEL_11;
        v12[1] = *v12;
        v26 = *(_QWORD *)(a3 + 32);
        while ( 1 )
        {
          if ( v26 == v18 )
            goto LABEL_9;
          v18 = *(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v18 )
            BUG();
          v27 = *(_WORD *)(v18 + 46);
          v28 = *(_QWORD *)v18;
          if ( (*(_QWORD *)v18 & 4) != 0 )
          {
            if ( (v27 & 4) != 0 )
            {
LABEL_70:
              v30 = (*(_QWORD *)(*(_QWORD *)(v18 + 16) + 8LL) >> 4) & 1LL;
              goto LABEL_32;
            }
          }
          else if ( (v27 & 4) != 0 )
          {
            while ( 1 )
            {
              v29 = v28 & 0xFFFFFFFFFFFFFFF8LL;
              v27 = *(_WORD *)(v29 + 46);
              v18 = v29;
              if ( (v27 & 4) == 0 )
                break;
              v28 = *(_QWORD *)v29;
            }
          }
          if ( (v27 & 8) == 0 )
            goto LABEL_70;
          LOBYTE(v30) = sub_1E15D00(v18, 0x10u, 1);
LABEL_32:
          if ( (_BYTE)v30 )
          {
            for ( k = *(_QWORD *)(*a1 + 272LL); (*(_BYTE *)(v18 + 46) & 4) != 0; v18 = *(_QWORD *)v18
                                                                                     & 0xFFFFFFFFFFFFFFF8LL )
              ;
            v32 = *(_QWORD *)(k + 368);
            v33 = *(_DWORD *)(k + 384);
            if ( v33 )
            {
              v34 = (v33 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
              v35 = (__int64 *)(v32 + 16LL * v34);
              v36 = *v35;
              if ( *v35 == v18 )
              {
LABEL_37:
                v15 = v35[1];
                v12[1] = v15;
                goto LABEL_10;
              }
              v62 = 1;
              while ( v36 != -8 )
              {
                v65 = v62 + 1;
                v34 = (v33 - 1) & (v62 + v34);
                v35 = (__int64 *)(v32 + 16LL * v34);
                v36 = *v35;
                if ( v18 == *v35 )
                  goto LABEL_37;
                v62 = v65;
              }
            }
            v35 = (__int64 *)(v32 + 16LL * v33);
            goto LABEL_37;
          }
        }
      }
      v63 = 1;
      while ( v25 != -8 )
      {
        v64 = v63 + 1;
        v23 = (v22 - 1) & (v63 + v23);
        v24 = (__int64 *)(v21 + 16LL * v23);
        v25 = *v24;
        if ( *v24 == v19 )
          goto LABEL_20;
        v63 = v64;
      }
    }
    v24 = (__int64 *)(v21 + 16 * v22);
    goto LABEL_20;
  }
LABEL_9:
  v15 = v12[1];
LABEL_10:
  if ( (v15 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_11;
  v37 = v71;
  v38 = 8LL * (unsigned int)i;
  v67 = &v71[v38];
  v39 = v38 >> 3;
  v40 = v38 >> 5;
  if ( !v40 )
  {
LABEL_62:
    if ( v39 != 2 )
    {
      if ( v39 != 3 )
      {
        if ( v39 != 1 )
          goto LABEL_11;
        goto LABEL_65;
      }
      v58 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 272LL) + 392LL) + 16LL * *(unsigned int *)(*v37 + 48LL));
      v59 = (__int64 *)sub_1DB3C70((__int64 *)a2, v58);
      if ( v59 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
        && (*(_DWORD *)((*v59 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v59 >> 1) & 3)) <= (*(_DWORD *)((v58 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                 | (unsigned int)(v58 >> 1)
                                                                                                 & 3) )
      {
        goto LABEL_42;
      }
      ++v37;
    }
    v60 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 272LL) + 392LL) + 16LL * *(unsigned int *)(*v37 + 48LL));
    v61 = (__int64 *)sub_1DB3C70((__int64 *)a2, v60);
    if ( v61 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
      && (*(_DWORD *)((*v61 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v61 >> 1) & 3)) <= (*(_DWORD *)((v60 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(v60 >> 1)
                                                                                               & 3) )
    {
      goto LABEL_42;
    }
    ++v37;
LABEL_65:
    v56 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 272LL) + 392LL) + 16LL * *(unsigned int *)(*v37 + 48LL));
    v57 = (__int64 *)sub_1DB3C70((__int64 *)a2, v56);
    if ( v57 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
      && (*(_DWORD *)((*v57 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v57 >> 1) & 3)) <= (*(_DWORD *)((v56 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(v56 >> 1)
                                                                                               & 3) )
    {
      goto LABEL_42;
    }
LABEL_11:
    v16 = *v12;
    goto LABEL_12;
  }
  v70 = &v71[32 * v40];
  while ( 1 )
  {
    v41 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 272LL) + 392LL) + 16LL * *(unsigned int *)(*v37 + 48LL));
    v42 = (__int64 *)sub_1DB3C70((__int64 *)a2, v41);
    if ( v42 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
      && (*(_DWORD *)((*v42 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v42 >> 1) & 3) <= (*(_DWORD *)((v41 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v41 >> 1)
                                                                                             & 3) )
    {
      break;
    }
    v50 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 272LL) + 392LL) + 16LL * *(unsigned int *)(v37[1] + 48LL));
    v51 = (__int64 *)sub_1DB3C70((__int64 *)a2, v50);
    if ( v51 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
      && (*(_DWORD *)((*v51 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v51 >> 1) & 3) <= (*(_DWORD *)((v50 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v50 >> 1)
                                                                                             & 3) )
    {
      ++v37;
      break;
    }
    v52 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 272LL) + 392LL) + 16LL * *(unsigned int *)(v37[2] + 48LL));
    v53 = (__int64 *)sub_1DB3C70((__int64 *)a2, v52);
    if ( v53 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
      && (*(_DWORD *)((*v53 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v53 >> 1) & 3) <= (*(_DWORD *)((v52 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v52 >> 1)
                                                                                             & 3) )
    {
      v37 += 2;
      break;
    }
    v54 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 272LL) + 392LL) + 16LL * *(unsigned int *)(v37[3] + 48LL));
    v55 = (__int64 *)sub_1DB3C70((__int64 *)a2, v54);
    if ( v55 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
      && (*(_DWORD *)((*v55 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v55 >> 1) & 3) <= (*(_DWORD *)((v54 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v54 >> 1)
                                                                                             & 3) )
    {
      v37 += 3;
      break;
    }
    v37 += 4;
    if ( v70 == v37 )
    {
      v39 = v67 - v37;
      goto LABEL_62;
    }
  }
LABEL_42:
  if ( v67 == v37 )
    goto LABEL_11;
  v43 = v69 & 0xFFFFFFFFFFFFFFF8LL;
  v44 = (v69 >> 1) & 3;
  v45 = ((v69 >> 1) & 3) != 0 ? v43 | (2LL * ((int)v44 - 1)) : *(_QWORD *)v43 & 0xFFFFFFFFFFFFFFF8LL | 6;
  v46 = (__int64 *)sub_1DB3C70((__int64 *)a2, v45);
  if ( v46 == (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8)) )
    goto LABEL_11;
  if ( (*(_DWORD *)((*v46 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v46 >> 1) & 3)) > (*(_DWORD *)((v45 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                            | (unsigned int)(v45 >> 1)
                                                                                            & 3) )
    goto LABEL_11;
  v47 = v46[2];
  if ( !v47 )
    goto LABEL_11;
  v48 = *(_QWORD *)(v47 + 8);
  v49 = *(_DWORD *)((v48 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  if ( v49 >= *(_DWORD *)((v12[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
    && (unsigned __int64)(v49 | (v48 >> 1) & 3) < (*(_DWORD *)((v69 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)v44) )
  {
    goto LABEL_11;
  }
  v16 = v12[1];
LABEL_12:
  if ( v71 != v73 )
    _libc_free((unsigned __int64)v71);
  return v16;
}
