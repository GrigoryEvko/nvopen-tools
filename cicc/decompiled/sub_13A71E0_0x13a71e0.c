// Function: sub_13A71E0
// Address: 0x13a71e0
//
__int64 __fastcall sub_13A71E0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, _QWORD *a5, __int64 *a6)
{
  int v9; // ecx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  int v14; // r8d
  int v15; // r9d
  unsigned __int64 v16; // rdi
  unsigned int v17; // r12d
  unsigned __int64 v18; // r14
  char v19; // bl
  unsigned __int64 *v20; // r13
  char v21; // r15
  __int64 v23; // rax
  _QWORD *v24; // r15
  __int64 v25; // rax
  size_t v26; // r10
  __int64 v27; // rdx
  unsigned int v28; // ecx
  void *v29; // r9
  int v30; // esi
  __int64 v31; // rdx
  __int64 v32; // rax
  _QWORD *v33; // r15
  __int64 v34; // rax
  size_t v35; // r10
  __int64 v36; // r9
  unsigned int v37; // edx
  void *v38; // r8
  _QWORD *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // r13
  __int64 v43; // r14
  int v44; // eax
  unsigned int v45; // edx
  _QWORD *v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rdx
  unsigned int v49; // edx
  _QWORD *v50; // rax
  int v51; // r13d
  __int64 v52; // rbx
  __int64 v53; // r15
  __int64 v54; // rax
  __int64 v55; // rax
  size_t v56; // [rsp+0h] [rbp-70h]
  size_t v57; // [rsp+0h] [rbp-70h]
  unsigned int v58; // [rsp+8h] [rbp-68h]
  unsigned int v59; // [rsp+8h] [rbp-68h]
  __int64 v60; // [rsp+8h] [rbp-68h]
  __int64 v61; // [rsp+8h] [rbp-68h]
  int v62; // [rsp+10h] [rbp-60h]
  unsigned int v63; // [rsp+10h] [rbp-60h]
  unsigned int v64; // [rsp+10h] [rbp-60h]
  int n; // [rsp+18h] [rbp-58h]
  size_t na; // [rsp+18h] [rbp-58h]
  size_t nb; // [rsp+18h] [rbp-58h]
  size_t nc; // [rsp+18h] [rbp-58h]
  int v70; // [rsp+20h] [rbp-50h]
  _QWORD *v72; // [rsp+28h] [rbp-48h]
  unsigned __int64 v73; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 v74[7]; // [rsp+38h] [rbp-38h] BYREF

  v9 = *(_DWORD *)(a1 + 40);
  v10 = (unsigned int)(v9 + 1);
  v73 = 1;
  if ( (unsigned int)v10 <= 0x39 )
  {
    v11 = (v10 << 58) | 1;
    v73 = v11;
LABEL_3:
    v74[0] = v11;
    goto LABEL_4;
  }
  v62 = v9 + 1;
  n = v9;
  v23 = sub_22077B0(24);
  v24 = (_QWORD *)v23;
  if ( v23 )
  {
    *(_QWORD *)v23 = 0;
    *(_QWORD *)(v23 + 8) = 0;
    *(_DWORD *)(v23 + 16) = v62;
    v58 = (unsigned int)(n + 64) >> 6;
    v25 = malloc(8LL * v58);
    v26 = 8LL * v58;
    v27 = v58;
    v28 = v58;
    v29 = (void *)v25;
    if ( !v25 )
    {
      if ( 8LL * v58 || (v55 = malloc(1u), v29 = 0, v28 = (unsigned int)(n + 64) >> 6, v27 = v58, v26 = 0, !v55) )
      {
        v57 = v26;
        v61 = v27;
        v64 = v28;
        nc = (size_t)v29;
        sub_16BD1C0("Allocation failed");
        v29 = (void *)nc;
        v28 = v64;
        v27 = v61;
        v26 = v57;
      }
      else
      {
        v29 = (void *)v55;
      }
    }
    *v24 = v29;
    v24[1] = v27;
    if ( v28 )
      memset(v29, 0, v26);
  }
  v30 = *(_DWORD *)(a1 + 40);
  v73 = (unsigned __int64)v24;
  v74[0] = 1;
  v31 = (unsigned int)(v30 + 1);
  if ( (unsigned int)v31 <= 0x39 )
  {
    v11 = (v31 << 58) | 1;
    goto LABEL_3;
  }
  v32 = sub_22077B0(24);
  v33 = (_QWORD *)v32;
  if ( v32 )
  {
    *(_QWORD *)v32 = 0;
    *(_QWORD *)(v32 + 8) = 0;
    *(_DWORD *)(v32 + 16) = v30 + 1;
    v59 = (unsigned int)(v30 + 64) >> 6;
    v34 = malloc(8LL * v59);
    v35 = 8LL * v59;
    v36 = v59;
    v37 = v59;
    v38 = (void *)v34;
    if ( !v34 )
    {
      if ( 8LL * v59 || (v54 = malloc(1u), v38 = 0, v37 = (unsigned int)(v30 + 64) >> 6, v36 = v59, v35 = 0, !v54) )
      {
        v56 = v35;
        v60 = v36;
        v63 = v37;
        nb = (size_t)v38;
        sub_16BD1C0("Allocation failed");
        v38 = (void *)nb;
        v37 = v63;
        v36 = v60;
        v35 = v56;
      }
      else
      {
        v38 = (void *)v54;
      }
    }
    *v33 = v38;
    v33[1] = v36;
    if ( v37 )
      memset(v38, 0, v35);
  }
  v74[0] = (unsigned __int64)v33;
LABEL_4:
  if ( !(unsigned __int8)sub_13A6F00(a1, a2, a3, &v73) || !(unsigned __int8)sub_13A7070(a1, a4, a5, v74) )
  {
    v18 = v74[0];
    v17 = 4;
    v19 = v74[0] & 1;
    goto LABEL_17;
  }
  sub_13A4D00(a6, (__int64 *)&v73);
  sub_13A5430((unsigned __int64 *)a6, v74, v12, v13, v14, v15);
  v16 = *a6;
  if ( (*a6 & 1) != 0 )
  {
    v17 = sub_39FAC40(~(-1LL << (v16 >> 58)) & (v16 >> 1));
  }
  else
  {
    v17 = (unsigned int)(*(_DWORD *)(v16 + 16) + 63) >> 6;
    if ( !v17 )
    {
      v18 = v74[0];
      v19 = v74[0] & 1;
      goto LABEL_17;
    }
    v39 = *(_QWORD **)v16;
    v40 = v17 - 1;
    v17 = 0;
    v41 = *(_QWORD *)v16 + 8LL;
    v42 = v41 + 8 * v40;
    while ( 1 )
    {
      v17 += sub_39FAC40(*v39);
      v39 = (_QWORD *)v41;
      if ( v42 == v41 )
        break;
      v41 += 8;
    }
  }
  v18 = v74[0];
  v19 = v74[0] & 1;
  if ( v17 < 2 )
    goto LABEL_17;
  if ( v17 != 2 )
  {
    v17 = 3;
    if ( v19 )
      goto LABEL_11;
LABEL_18:
    if ( v18 )
    {
      _libc_free(*(_QWORD *)v18);
      j_j___libc_free_0(v18, 24);
    }
    goto LABEL_11;
  }
  v20 = (unsigned __int64 *)v73;
  v21 = v73 & 1;
  if ( (v73 & 1) != 0 )
  {
    v70 = sub_39FAC40((v73 >> 1) & ~(-1LL << (v73 >> 58)));
    goto LABEL_37;
  }
  v45 = (unsigned int)(*(_DWORD *)(v73 + 16) + 63) >> 6;
  if ( !v45 )
  {
LABEL_17:
    if ( v19 )
      goto LABEL_11;
    goto LABEL_18;
  }
  v46 = *(_QWORD **)v73;
  v47 = v45 - 1;
  v70 = 0;
  v48 = *(_QWORD *)v73 + 8LL;
  na = v48 + 8 * v47;
  while ( 1 )
  {
    v72 = (_QWORD *)v48;
    v70 += sub_39FAC40(*v46);
    v46 = v72;
    if ( (_QWORD *)na == v72 )
      break;
    v48 = (__int64)(v72 + 1);
  }
LABEL_37:
  if ( !v70 )
    goto LABEL_17;
  if ( !v19 )
  {
    v49 = (unsigned int)(*(_DWORD *)(v18 + 16) + 63) >> 6;
    if ( v49 )
    {
      v50 = *(_QWORD **)v18;
      v51 = 0;
      v52 = *(_QWORD *)v18 + 8LL;
      v53 = v52 + 8LL * (v49 - 1);
      while ( 1 )
      {
        v51 += sub_39FAC40(*v50);
        v50 = (_QWORD *)v52;
        if ( v53 == v52 )
          break;
        v52 += 8;
      }
      if ( v51 && (v51 != 1 || v70 != 1) )
        v17 = 3;
    }
    goto LABEL_18;
  }
  v43 = ~(-1LL << (v18 >> 58)) & (v18 >> 1);
  v44 = sub_39FAC40(v43);
  if ( !v43 )
    goto LABEL_12;
  if ( v70 != 1 || v44 != 1 )
    v17 = 3;
LABEL_11:
  v20 = (unsigned __int64 *)v73;
  v21 = v73 & 1;
LABEL_12:
  if ( !v21 && v20 )
  {
    _libc_free(*v20);
    j_j___libc_free_0(v20, 24);
  }
  return v17;
}
