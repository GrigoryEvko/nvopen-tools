// Function: sub_2E13970
// Address: 0x2e13970
//
__int64 __fastcall sub_2E13970(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned int *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 *v17; // r14
  __int64 v18; // rdx
  __int64 *v19; // r13
  __int64 v20; // rsi
  __int64 *v21; // rdi
  __int64 v22; // rdx
  unsigned int v23; // edi
  __int64 v24; // rsi
  unsigned __int64 v25; // r10
  __int64 *v26; // rsi
  unsigned int v27; // r10d
  __int64 v28; // rax
  int v29; // esi
  int v30; // eax
  __int64 v31; // rdi
  int v32; // edx
  unsigned int v33; // eax
  __int64 v34; // r9
  unsigned int v35; // edi
  __int64 v36; // r11
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rdi
  __int64 v41; // [rsp-B0h] [rbp-B0h]
  int v42; // [rsp-A8h] [rbp-A8h]
  __int64 v43; // [rsp-A0h] [rbp-A0h]
  unsigned int v44; // [rsp-A0h] [rbp-A0h]
  __int64 *v45; // [rsp-98h] [rbp-98h]
  __int64 v47; // [rsp-90h] [rbp-90h]
  __int64 v48; // [rsp-90h] [rbp-90h]
  unsigned __int8 v49; // [rsp-79h] [rbp-79h] BYREF
  __int64 v50; // [rsp-78h] [rbp-78h] BYREF
  __int64 v51; // [rsp-70h] [rbp-70h]
  __int64 v52; // [rsp-68h] [rbp-68h] BYREF
  int v53; // [rsp-60h] [rbp-60h]
  _QWORD v54[11]; // [rsp-58h] [rbp-58h] BYREF

  v3 = *(unsigned int *)(a2 + 8);
  if ( !(_DWORD)v3 )
    return 0;
  v7 = *(__int64 **)a2;
  v50 = 0;
  v51 = 0;
  v45 = &v7[3 * v3];
  v8 = sub_2E13500(a1, a2);
  v10 = a3;
  if ( v8 )
  {
    v11 = *(_QWORD *)(a1 + 184);
    v12 = (unsigned int *)(*(_QWORD *)(a1 + 344) + 8LL * *(unsigned int *)(v8 + 24));
    v13 = v12[1];
    v14 = 8LL * *v12;
    v51 = v13;
    v15 = v14 + v11;
    v50 = *(_QWORD *)(a1 + 264) + v14;
  }
  else
  {
    v39 = *(_QWORD *)(a1 + 264);
    v40 = *(unsigned int *)(a1 + 272);
    v15 = *(_QWORD *)(a1 + 184);
    v13 = *(unsigned int *)(a1 + 192);
    v50 = v39;
    v51 = v40;
  }
  v16 = 8 * v13;
  v17 = (__int64 *)(v15 + v16);
  v18 = v16 >> 3;
  if ( v16 )
  {
    v19 = (__int64 *)v15;
    do
    {
      while ( 1 )
      {
        v20 = v18 >> 1;
        v21 = &v19[v18 >> 1];
        if ( (*(_DWORD *)((*v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v21 >> 1) & 3) >= (*(_DWORD *)((*v7 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                 | (unsigned int)(*v7 >> 1)
                                                                                                 & 3) )
          break;
        v19 = v21 + 1;
        v18 = v18 - v20 - 1;
        if ( v18 <= 0 )
          goto LABEL_10;
      }
      v18 >>= 1;
    }
    while ( v20 > 0 );
  }
  else
  {
    v19 = (__int64 *)v15;
  }
LABEL_10:
  if ( v17 == v19 )
    return 0;
  v54[2] = a1;
  v54[0] = &v49;
  v54[3] = &v50;
  v49 = 0;
  v54[1] = a3;
  v22 = *v19;
  v23 = *(_DWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  while ( 1 )
  {
    while ( 1 )
    {
      v24 = v7[1];
      v25 = v24 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (unsigned __int64)((v22 >> 1) & 3 | v23) >= (*(_DWORD *)((v24 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                      | (unsigned int)(v24 >> 1) & 3) )
        break;
      v47 = v15;
      sub_2E10440((__int64)v54, ((__int64)v19 - v15) >> 3, v22, v10, v15, v9);
      v15 = v47;
      if ( v19 + 1 == v17 )
        return v49;
      v22 = v19[1];
      ++v19;
      v23 = *(_DWORD *)((v22 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    }
    if ( v24 == v22 )
    {
      v10 = *(_QWORD *)(v25 + 16);
      if ( v10 )
      {
        if ( *(_WORD *)(v10 + 68) == 32 )
        {
          v29 = *(_DWORD *)(a2 + 112);
          v52 = *(_QWORD *)(v25 + 16);
          v43 = v15;
          v48 = v10;
          v30 = sub_2E88FE0(v52);
          v10 = v48;
          v15 = v43;
          v31 = *(_QWORD *)(v52 + 32);
          v53 = v30 + *(unsigned __int8 *)(*(_QWORD *)(v48 + 16) + 9LL);
          v32 = *(_DWORD *)(v31 + 40LL * (unsigned int)(v53 + 2) + 24) + v53;
          if ( (*(_BYTE *)(v31 + 40LL * (unsigned int)(v32 + 7) + 24) & 2) == 0 )
          {
            v41 = v43;
            v44 = v32 + 9;
            v42 = *(_DWORD *)(v31 + 40LL * (unsigned int)(v53 + 2) + 24) + v53;
            v33 = sub_2FC8910(&v52);
            v15 = v41;
            v35 = v33;
            if ( v44 < v33 )
            {
              v10 = v48;
              v36 = *(_QWORD *)(v48 + 32);
              v37 = v36 + 40LL * v44;
              v38 = v36 + 40 * (v44 + (unsigned __int64)(v35 - v42 - 10)) + 40;
              while ( *(_BYTE *)v37 || v29 != *(_DWORD *)(v37 + 8) )
              {
                v37 += 40;
                if ( v38 == v37 )
                  goto LABEL_16;
              }
              sub_2E10440((__int64)v54, ((__int64)v19++ - v41) >> 3, v38, v48, v41, v34);
              v15 = v41;
            }
          }
        }
      }
    }
LABEL_16:
    v26 = v7 + 3;
    if ( v7 + 3 == v45 )
      return v49;
    if ( v19 == v17 )
      return v49;
    v22 = *v19;
    v23 = *(_DWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    v27 = v23 | (*v19 >> 1) & 3;
    if ( v27 > (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8) - 16) & 0xFFFFFFFFFFFFFFF8LL)
                          + 24)
              | (unsigned int)(*(__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8) - 16) >> 1) & 3) )
      return v49;
    if ( (*(_DWORD *)((v7[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v7[4] >> 1) & 3) < v27 )
    {
      do
      {
        v28 = v26[4];
        v26 += 3;
      }
      while ( v27 > (*(_DWORD *)((v28 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v28 >> 1) & 3) );
    }
    v9 = *(_DWORD *)((*v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v26 >> 1) & 3;
    while ( (unsigned __int64)(v23 | (v22 >> 1) & 3) < (unsigned int)v9 )
    {
      if ( ++v19 == v17 )
        return v49;
      v22 = *v19;
      v23 = *(_DWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    }
    v7 = v26;
  }
}
