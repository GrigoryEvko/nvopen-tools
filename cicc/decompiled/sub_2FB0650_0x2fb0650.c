// Function: sub_2FB0650
// Address: 0x2fb0650
//
__int64 __fastcall sub_2FB0650(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5)
{
  __int64 v5; // r9
  char v6; // r10
  __int64 v8; // r12
  __int64 *v9; // rbx
  __int64 v10; // rax
  __int64 *v11; // r14
  __int64 *i; // r11
  __int64 v13; // r13
  char v14; // al
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // r9
  char v22; // r10
  __int64 v23; // rbx
  __int64 v24; // rax
  unsigned __int64 v25; // r13
  __int64 v26; // r12
  char v27; // bl
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  __int64 v30; // rax
  unsigned __int64 v31; // rax
  _QWORD *v32; // r13
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rbx
  __int64 *v37; // rax
  unsigned __int64 v38; // rdi
  __int64 v39; // r13
  __int64 *v40; // rbx
  __int64 v41; // rdx
  unsigned __int64 v42; // rax
  __int64 v43; // rbx
  __int64 *v44; // rax
  __int64 v45; // rbx
  __int64 *v46; // rax
  __int64 v47; // rbx
  __int64 *v48; // rax
  __int64 v49; // rbx
  __int64 *v50; // rax
  int v51; // eax
  char v52; // al
  __int64 j; // r13
  __int64 v54; // rbx
  __int64 *v55; // rax
  __int64 v56; // rbx
  __int64 *v57; // rdi
  __int64 v58; // rsi
  __int64 v59; // [rsp+0h] [rbp-80h]
  __int64 v60; // [rsp+0h] [rbp-80h]
  __int64 *v61; // [rsp+8h] [rbp-78h]
  __int64 *v62; // [rsp+8h] [rbp-78h]
  char v63; // [rsp+10h] [rbp-70h]
  __int64 v64; // [rsp+10h] [rbp-70h]
  char v65; // [rsp+10h] [rbp-70h]
  _QWORD *v66; // [rsp+10h] [rbp-70h]
  char v67; // [rsp+10h] [rbp-70h]
  __int64 v68; // [rsp+18h] [rbp-68h]
  __int64 v69; // [rsp+28h] [rbp-58h]
  char v70; // [rsp+28h] [rbp-58h]
  _QWORD *v71; // [rsp+28h] [rbp-58h]
  _BYTE *v72; // [rsp+30h] [rbp-50h] BYREF
  __int64 v73; // [rsp+38h] [rbp-48h]
  _BYTE v74[64]; // [rsp+40h] [rbp-40h] BYREF

  v5 = a3;
  v6 = 0;
  v8 = a2;
  v9 = *(__int64 **)(a3 + 112);
  v10 = 16LL * *(unsigned int *)(a3 + 24);
  v11 = (__int64 *)(v10 + a1[1]);
  v68 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 32LL) + 152LL) + v10 + 8);
  v72 = v74;
  v73 = 0x100000000LL;
  for ( i = &v9[*(unsigned int *)(a3 + 120)]; i != v9; ++v9 )
  {
    while ( 1 )
    {
      v13 = *v9;
      v14 = *(_BYTE *)(*v9 + 216);
      if ( !v14 )
        break;
      v15 = (unsigned int)v73;
      a5 = (unsigned int)v73 + 1LL;
      if ( a5 > HIDWORD(v73) )
      {
        v59 = v5;
        v61 = i;
        v65 = *(_BYTE *)(*v9 + 216);
        sub_C8D5F0((__int64)&v72, v74, (unsigned int)v73 + 1LL, 8u, a5, v5);
        v15 = (unsigned int)v73;
        v5 = v59;
        i = v61;
        v14 = v65;
      }
      ++v9;
      v6 = v14;
      *(_QWORD *)&v72[8 * v15] = v13;
      LODWORD(v73) = v73 + 1;
      if ( i == v9 )
        goto LABEL_9;
    }
    if ( *(_BYTE *)(v13 + 262) )
    {
      v18 = (unsigned int)v73;
      v19 = (unsigned int)v73 + 1LL;
      if ( v19 > HIDWORD(v73) )
      {
        v60 = v5;
        v62 = i;
        v67 = v6;
        sub_C8D5F0((__int64)&v72, v74, v19, 8u, a5, v5);
        v18 = (unsigned int)v73;
        v5 = v60;
        i = v62;
        v6 = v67;
      }
      *(_QWORD *)&v72[8 * v18] = v13;
      LODWORD(v73) = v73 + 1;
    }
  }
LABEL_9:
  if ( (*v11 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    v63 = v6;
    v69 = v5;
    v20 = sub_2E313E0(v5);
    v21 = v69;
    v22 = v63;
    v23 = v69 + 48;
    if ( v69 + 48 == v20 )
    {
      *v11 = v68;
    }
    else
    {
      v64 = v69;
      v70 = v22;
      v24 = sub_2DF8360(*(_QWORD *)(*a1 + 32LL), v20, 0);
      v21 = v64;
      v22 = v70;
      *v11 = v24;
    }
    if ( !(_DWORD)v73 )
      goto LABEL_11;
    v25 = *(_QWORD *)(v21 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v25 )
      BUG();
    if ( (*(_QWORD *)v25 & 4) == 0 && (*(_BYTE *)(v25 + 44) & 4) != 0 )
    {
      for ( j = *(_QWORD *)v25; ; j = *(_QWORD *)v25 )
      {
        v25 = j & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v25 + 44) & 4) == 0 )
          break;
      }
    }
    if ( v23 != v25 )
    {
      v26 = v23;
      v27 = v22;
      while ( 1 )
      {
        if ( v27 )
        {
          v51 = *(_DWORD *)(v25 + 44);
          if ( (v51 & 4) != 0 || (v51 & 8) == 0 )
            v52 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v25 + 16) + 24LL) >> 7;
          else
            v52 = sub_2E88A90(v25, 128, 1);
          if ( v52 )
            break;
        }
        if ( *(_WORD *)(v25 + 68) == 2 )
          break;
        v28 = (_QWORD *)(*(_QWORD *)v25 & 0xFFFFFFFFFFFFFFF8LL);
        v29 = v28;
        if ( !v28 )
          BUG();
        v25 = *(_QWORD *)v25 & 0xFFFFFFFFFFFFFFF8LL;
        v30 = *v28;
        if ( (v30 & 4) == 0 && (*((_BYTE *)v29 + 44) & 4) != 0 )
        {
          while ( 1 )
          {
            v31 = v30 & 0xFFFFFFFFFFFFFFF8LL;
            v25 = v31;
            if ( (*(_BYTE *)(v31 + 44) & 4) == 0 )
              break;
            v30 = *(_QWORD *)v31;
          }
        }
        if ( v26 == v25 )
        {
          v8 = a2;
          goto LABEL_10;
        }
      }
      v8 = a2;
      v11[1] = sub_2DF8360(*(_QWORD *)(*a1 + 32LL), v25, 0);
    }
  }
LABEL_10:
  if ( (v11[1] & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
LABEL_11:
    v16 = *v11;
    goto LABEL_12;
  }
  v32 = v72;
  v33 = 8LL * (unsigned int)v73;
  v66 = &v72[v33];
  v34 = v33 >> 3;
  v35 = v33 >> 5;
  if ( v35 )
  {
    v71 = &v72[32 * v35];
    while ( 1 )
    {
      v36 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 32LL) + 152LL) + 16LL * *(unsigned int *)(*v32 + 24LL));
      v37 = (__int64 *)sub_2E09D00((__int64 *)v8, v36);
      if ( v37 != (__int64 *)(*(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8))
        && (*(_DWORD *)((*v37 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v37 >> 1) & 3) <= (*(_DWORD *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(v36 >> 1)
                                                                                               & 3) )
      {
        goto LABEL_39;
      }
      v43 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 32LL) + 152LL) + 16LL * *(unsigned int *)(v32[1] + 24LL));
      v44 = (__int64 *)sub_2E09D00((__int64 *)v8, v43);
      if ( v44 != (__int64 *)(*(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8))
        && (*(_DWORD *)((*v44 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v44 >> 1) & 3) <= (*(_DWORD *)((v43 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(v43 >> 1)
                                                                                               & 3) )
      {
        ++v32;
        goto LABEL_39;
      }
      v45 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 32LL) + 152LL) + 16LL * *(unsigned int *)(v32[2] + 24LL));
      v46 = (__int64 *)sub_2E09D00((__int64 *)v8, v45);
      if ( v46 != (__int64 *)(*(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8))
        && (*(_DWORD *)((*v46 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v46 >> 1) & 3) <= (*(_DWORD *)((v45 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(v45 >> 1)
                                                                                               & 3) )
      {
        v32 += 2;
        goto LABEL_39;
      }
      v47 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 32LL) + 152LL) + 16LL * *(unsigned int *)(v32[3] + 24LL));
      v48 = (__int64 *)sub_2E09D00((__int64 *)v8, v47);
      if ( v48 != (__int64 *)(*(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8))
        && (*(_DWORD *)((*v48 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v48 >> 1) & 3) <= (*(_DWORD *)((v47 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(v47 >> 1)
                                                                                               & 3) )
      {
        v32 += 3;
        goto LABEL_39;
      }
      v32 += 4;
      if ( v32 == v71 )
      {
        v34 = v66 - v32;
        break;
      }
    }
  }
  if ( v34 == 2 )
    goto LABEL_82;
  if ( v34 == 3 )
  {
    v54 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 32LL) + 152LL) + 16LL * *(unsigned int *)(*v32 + 24LL));
    v55 = (__int64 *)sub_2E09D00((__int64 *)v8, v54);
    if ( v55 != (__int64 *)(*(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8))
      && (*(_DWORD *)((*v55 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v55 >> 1) & 3)) <= (*(_DWORD *)((v54 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(v54 >> 1)
                                                                                               & 3) )
    {
      goto LABEL_39;
    }
    ++v32;
LABEL_82:
    v56 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 32LL) + 152LL) + 16LL * *(unsigned int *)(*v32 + 24LL));
    v57 = (__int64 *)sub_2E09D00((__int64 *)v8, v56);
    if ( v57 != (__int64 *)(*(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8)) && sub_2DF8330(v57, v56) )
      goto LABEL_39;
    ++v32;
    goto LABEL_63;
  }
  if ( v34 != 1 )
    goto LABEL_11;
LABEL_63:
  v49 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 32LL) + 152LL) + 16LL * *(unsigned int *)(*v32 + 24LL));
  v50 = (__int64 *)sub_2E09D00((__int64 *)v8, v49);
  if ( v50 == (__int64 *)(*(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8))
    || (*(_DWORD *)((*v50 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v50 >> 1) & 3)) > (*(_DWORD *)((v49 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                            | (unsigned int)(v49 >> 1)
                                                                                            & 3) )
  {
    goto LABEL_11;
  }
LABEL_39:
  if ( v66 == v32 )
    goto LABEL_11;
  v38 = v68 & 0xFFFFFFFFFFFFFFF8LL;
  v39 = ((v68 >> 1) & 3) != 0 ? v38 | (2LL * (int)(((v68 >> 1) & 3) - 1)) : *(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL | 6;
  v40 = (__int64 *)sub_2E09D00((__int64 *)v8, v39);
  if ( v40 == (__int64 *)(*(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8)) )
    goto LABEL_11;
  if ( !sub_2DF8330(v40, v39) )
    goto LABEL_11;
  v41 = v40[2];
  if ( !v41 )
    goto LABEL_11;
  v16 = v11[1];
  v42 = *(_QWORD *)(v41 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v42 != (v16 & 0xFFFFFFFFFFFFFFF8LL) || (v58 = *(_QWORD *)(v42 + 16)) == 0 || *(_WORD *)(v58 + 68) != 32 )
  {
    if ( *(_DWORD *)(v42 + 24) >= *(_DWORD *)((v16 & 0xFFFFFFFFFFFFFFF8LL) + 24)
      && sub_2DF8300((__int64 *)(v41 + 8), v68) )
    {
      goto LABEL_11;
    }
    v16 = v11[1];
  }
LABEL_12:
  if ( v72 != v74 )
    _libc_free((unsigned __int64)v72);
  return v16;
}
