// Function: sub_327F470
// Address: 0x327f470
//
__int64 __fastcall sub_327F470(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rax
  __int64 v9; // rax
  void *v10; // rax
  int v11; // r9d
  __int64 v12; // rdx
  int v13; // r9d
  __int64 v14; // rax
  __int64 v15; // r14
  __int128 v16; // rax
  int v17; // r9d
  __int128 v18; // rax
  __int64 v19; // rax
  void *v20; // rax
  int v21; // r9d
  __int64 v22; // rdx
  __int64 v23; // r14
  __int128 v24; // rax
  int v25; // r9d
  __int64 v26; // r14
  __int128 v27; // rax
  int v28; // r9d
  __int64 v29; // rdi
  bool v30; // al
  _QWORD *i; // rdi
  __int64 v32; // rdi
  int v33; // r9d
  __int64 v34; // rax
  __int64 v35; // rdi
  bool v36; // al
  __int64 v37; // rax
  _QWORD *j; // r14
  _QWORD *k; // rdi
  __int64 v40; // rdi
  _QWORD *m; // r14
  __int128 v42; // [rsp-F8h] [rbp-F8h]
  __int128 v43; // [rsp-F8h] [rbp-F8h]
  __int128 v44; // [rsp-E8h] [rbp-E8h]
  __int128 v45; // [rsp-E8h] [rbp-E8h]
  __int128 v46; // [rsp-E8h] [rbp-E8h]
  __int128 v47; // [rsp-D8h] [rbp-D8h]
  __int128 v48; // [rsp-D8h] [rbp-D8h]
  __int128 v49; // [rsp-D8h] [rbp-D8h]
  __int64 v50; // [rsp-C0h] [rbp-C0h]
  __int64 v51; // [rsp-C0h] [rbp-C0h]
  __int64 v52; // [rsp-B8h] [rbp-B8h]
  __int64 v53; // [rsp-B8h] [rbp-B8h]
  __int64 v54; // [rsp-B0h] [rbp-B0h]
  __int64 v55; // [rsp-B0h] [rbp-B0h]
  _DWORD *v56; // [rsp-A8h] [rbp-A8h]
  _DWORD *v57; // [rsp-A8h] [rbp-A8h]
  void *v58; // [rsp-A0h] [rbp-A0h]
  void *v59; // [rsp-A0h] [rbp-A0h]
  bool v60; // [rsp-91h] [rbp-91h]
  bool v61; // [rsp-91h] [rbp-91h]
  __int64 v62; // [rsp-90h] [rbp-90h]
  __int64 v63; // [rsp-90h] [rbp-90h]
  __int64 v64; // [rsp-90h] [rbp-90h]
  __int64 v65; // [rsp-90h] [rbp-90h]
  __int128 v66; // [rsp-88h] [rbp-88h]
  void *v67; // [rsp-78h] [rbp-78h] BYREF
  _QWORD *v68; // [rsp-70h] [rbp-70h]
  __int64 v69[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( *(_DWORD *)(a2 + 24) != 97 )
    return 0;
  if ( !**(_BYTE **)a1 )
  {
    v8 = *(_QWORD *)(a2 + 56);
    if ( !v8 || *(_QWORD *)(v8 + 32) )
      return 0;
  }
  v9 = sub_33E1790(**(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), 1);
  v50 = v9;
  if ( v9 )
  {
    v62 = *(_QWORD *)(v9 + 96);
    v56 = sub_C33320();
    sub_C3B1B0((__int64)v69, 1.0);
    sub_C407B0(&v67, v69, v56);
    sub_C338F0((__int64)v69);
    sub_C41640((__int64 *)&v67, *(_DWORD **)(v62 + 24), 1, (bool *)v69);
    v52 = (__int64)v67;
    v54 = *(_QWORD *)(v62 + 24);
    v10 = sub_C33340();
    v60 = 0;
    v12 = v52;
    v58 = v10;
    if ( v54 == v52 )
    {
      v29 = v62 + 24;
      if ( (void *)v52 == v10 )
        v30 = sub_C3E590(v29, (__int64)&v67);
      else
        v30 = sub_C33D00(v29, (__int64)&v67);
      v12 = (__int64)v67;
      v60 = v30;
    }
    if ( v58 == (void *)v12 )
    {
      if ( v68 )
      {
        for ( i = &v68[3 * *(v68 - 1)]; v68 != i; i -= 3 )
          sub_91D830(i - 3);
        j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v67);
    }
    if ( v60 )
    {
      v26 = **(_QWORD **)(a1 + 8);
      *(_QWORD *)&v27 = sub_33FAF80(
                          v26,
                          244,
                          *(_QWORD *)(a1 + 24),
                          **(_DWORD **)(a1 + 32),
                          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
                          v11,
                          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
      *((_QWORD *)&v45 + 1) = a4;
      *(_QWORD *)&v45 = a3;
      *((_QWORD *)&v43 + 1) = a4;
      *(_QWORD *)&v43 = a3;
      return sub_340F900(
               v26,
               **(_DWORD **)(a1 + 16),
               *(_QWORD *)(a1 + 24),
               **(_DWORD **)(a1 + 32),
               *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
               v28,
               v27,
               v43,
               v45);
    }
    v63 = *(_QWORD *)(v50 + 96);
    sub_C3B1B0((__int64)v69, -1.0);
    sub_C407B0(&v67, v69, v56);
    sub_C338F0((__int64)v69);
    sub_C41640((__int64 *)&v67, *(_DWORD **)(v63 + 24), 1, (bool *)v69);
    v14 = (__int64)v67;
    if ( *(void **)(v63 + 24) == v67 )
    {
      v32 = v63 + 24;
      if ( v67 == v58 )
        v60 = sub_C3E590(v32, (__int64)&v67);
      else
        v60 = sub_C33D00(v32, (__int64)&v67);
      v14 = (__int64)v67;
    }
    if ( v58 == (void *)v14 )
    {
      if ( v68 )
      {
        v37 = 3LL * *(v68 - 1);
        for ( j = &v68[v37]; v68 != j; sub_91D830(j) )
          j -= 3;
        j_j_j___libc_free_0_0((unsigned __int64)(j - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v67);
    }
    if ( v60 )
    {
      v15 = **(_QWORD **)(a1 + 8);
      *((_QWORD *)&v47 + 1) = a4;
      *(_QWORD *)&v47 = a3;
      *(_QWORD *)&v16 = sub_33FAF80(
                          v15,
                          244,
                          *(_QWORD *)(a1 + 24),
                          **(_DWORD **)(a1 + 32),
                          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
                          v13,
                          v47);
      v66 = v16;
      *(_QWORD *)&v18 = sub_33FAF80(
                          **(_QWORD **)(a1 + 8),
                          244,
                          *(_QWORD *)(a1 + 24),
                          **(_DWORD **)(a1 + 32),
                          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
                          v17,
                          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
      *((_QWORD *)&v44 + 1) = a4;
      *(_QWORD *)&v44 = a3;
      return sub_340F900(
               v15,
               **(_DWORD **)(a1 + 16),
               *(_QWORD *)(a1 + 24),
               **(_DWORD **)(a1 + 32),
               *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
               DWORD2(v18),
               v18,
               v44,
               v66);
    }
  }
  v19 = sub_33E1790(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL), 1);
  v55 = v19;
  if ( !v19 )
    return 0;
  v64 = *(_QWORD *)(v19 + 96);
  v57 = sub_C33320();
  sub_C3B1B0((__int64)v69, 1.0);
  sub_C407B0(&v67, v69, v57);
  sub_C338F0((__int64)v69);
  sub_C41640((__int64 *)&v67, *(_DWORD **)(v64 + 24), 1, (bool *)v69);
  v51 = (__int64)v67;
  v53 = *(_QWORD *)(v64 + 24);
  v20 = sub_C33340();
  v61 = 0;
  v22 = v51;
  v59 = v20;
  if ( v53 == v51 )
  {
    v35 = v64 + 24;
    if ( v20 == (void *)v51 )
      v36 = sub_C3E590(v35, (__int64)&v67);
    else
      v36 = sub_C33D00(v35, (__int64)&v67);
    v22 = (__int64)v67;
    v61 = v36;
  }
  if ( v59 == (void *)v22 )
  {
    if ( v68 )
    {
      for ( k = &v68[3 * *(v68 - 1)]; v68 != k; k -= 3 )
        sub_91D830(k - 3);
      j_j_j___libc_free_0_0((unsigned __int64)(k - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v67);
  }
  if ( v61 )
  {
    v23 = **(_QWORD **)(a1 + 8);
    *((_QWORD *)&v48 + 1) = a4;
    *(_QWORD *)&v48 = a3;
    *(_QWORD *)&v24 = sub_33FAF80(
                        v23,
                        244,
                        *(_QWORD *)(a1 + 24),
                        **(_DWORD **)(a1 + 32),
                        *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
                        v21,
                        v48);
    *((_QWORD *)&v42 + 1) = a4;
    *(_QWORD *)&v42 = a3;
    return sub_340F900(
             v23,
             **(_DWORD **)(a1 + 16),
             *(_QWORD *)(a1 + 24),
             **(_DWORD **)(a1 + 32),
             *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
             v25,
             *(_OWORD *)*(_QWORD *)(a2 + 40),
             v42,
             v24);
  }
  v65 = *(_QWORD *)(v55 + 96);
  sub_C3B1B0((__int64)v69, -1.0);
  sub_C407B0(&v67, v69, v57);
  sub_C338F0((__int64)v69);
  sub_C41640((__int64 *)&v67, *(_DWORD **)(v65 + 24), 1, (bool *)v69);
  v34 = (__int64)v67;
  if ( *(void **)(v65 + 24) == v67 )
  {
    v40 = v65 + 24;
    if ( v59 == v67 )
      v61 = sub_C3E590(v40, (__int64)&v67);
    else
      v61 = sub_C33D00(v40, (__int64)&v67);
    v34 = (__int64)v67;
  }
  if ( (void *)v34 == v59 )
  {
    if ( v68 )
    {
      for ( m = &v68[3 * *(v68 - 1)]; v68 != m; sub_91D830(m) )
        m -= 3;
      j_j_j___libc_free_0_0((unsigned __int64)(m - 1));
    }
  }
  else
  {
    sub_C338F0((__int64)&v67);
  }
  if ( !v61 )
    return 0;
  *((_QWORD *)&v49 + 1) = a4;
  *(_QWORD *)&v49 = a3;
  *((_QWORD *)&v46 + 1) = a4;
  *(_QWORD *)&v46 = a3;
  return sub_340F900(
           **(_QWORD **)(a1 + 8),
           **(_DWORD **)(a1 + 16),
           *(_QWORD *)(a1 + 24),
           **(_DWORD **)(a1 + 32),
           *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL),
           v33,
           *(_OWORD *)*(_QWORD *)(a2 + 40),
           v46,
           v49);
}
