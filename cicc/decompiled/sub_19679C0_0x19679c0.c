// Function: sub_19679C0
// Address: 0x19679c0
//
__int64 __fastcall sub_19679C0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v12; // eax
  unsigned int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 *v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 *v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  double v41; // xmm4_8
  double v42; // xmm5_8
  unsigned int v43; // eax
  __int64 *v44; // r12
  __int64 *v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  int v49; // ecx
  __int64 v50; // rax
  _QWORD *v51; // rdi
  _QWORD *i; // rax
  _QWORD *v53; // r13
  _QWORD *v54; // rax
  _QWORD *v55; // r12
  unsigned int v56; // edx
  _QWORD *v57; // r8
  unsigned int v58; // ecx
  unsigned int v59; // edx
  int v60; // edx
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rax
  int v63; // r13d
  __int64 v64; // r12
  _QWORD *v65; // rax
  __int64 v66; // rdx
  _QWORD *j; // rdx
  __int64 v68; // r14
  __int64 v69; // rdi
  __int64 v70; // rsi
  __int64 v71; // rdi
  int v72; // r15d
  __int64 v73; // rcx
  __int64 v74; // rax
  int v75; // edx
  __int64 v76; // r15
  _QWORD *v77; // rax
  _QWORD *v78; // rax
  _QWORD *v79; // r12
  _QWORD *v80; // rbx
  __int64 v81; // r13
  __int64 **v82; // [rsp+0h] [rbp-70h]
  __int64 v83; // [rsp+8h] [rbp-68h]
  __int64 v84; // [rsp+10h] [rbp-60h]
  _QWORD *v85; // [rsp+18h] [rbp-58h]
  __int64 *v86[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v87; // [rsp+30h] [rbp-40h]

  v12 = sub_1404700(a1, a2);
  if ( (_BYTE)v12 )
  {
    v49 = *(_DWORD *)(a1 + 176);
    if ( !v49 )
    {
      ++*(_QWORD *)(a1 + 160);
      goto LABEL_40;
    }
    v51 = *(_QWORD **)(a1 + 168);
    v53 = &v51[2 * *(unsigned int *)(a1 + 184)];
    if ( v51 == v53 )
      goto LABEL_51;
    v54 = *(_QWORD **)(a1 + 168);
    while ( 1 )
    {
      v55 = v54;
      if ( *v54 != -8 && *v54 != -16 )
        break;
      v54 += 2;
      if ( v53 == v54 )
        goto LABEL_51;
    }
    if ( v53 == v54 )
    {
LABEL_51:
      ++*(_QWORD *)(a1 + 160);
    }
    else
    {
      do
      {
        v68 = v55[1];
        if ( v68 )
        {
          sub_12D5E00(v55[1]);
          j_j___libc_free_0(v68, 72);
        }
        v55 += 2;
        if ( v55 == v53 )
          break;
        while ( *v55 == -8 || *v55 == -16 )
        {
          v55 += 2;
          if ( v53 == v55 )
            goto LABEL_72;
        }
      }
      while ( v53 != v55 );
LABEL_72:
      v49 = *(_DWORD *)(a1 + 176);
      ++*(_QWORD *)(a1 + 160);
      if ( !v49 )
      {
LABEL_40:
        if ( *(_DWORD *)(a1 + 180) )
        {
          v50 = *(unsigned int *)(a1 + 184);
          v51 = *(_QWORD **)(a1 + 168);
          if ( (unsigned int)v50 > 0x40 )
          {
            j___libc_free_0(v51);
            v13 = 0;
            *(_QWORD *)(a1 + 168) = 0;
            *(_QWORD *)(a1 + 176) = 0;
            *(_DWORD *)(a1 + 184) = 0;
            return v13;
          }
          goto LABEL_42;
        }
        return 0;
      }
      v51 = *(_QWORD **)(a1 + 168);
    }
    v56 = 4 * v49;
    v50 = *(unsigned int *)(a1 + 184);
    if ( (unsigned int)(4 * v49) < 0x40 )
      v56 = 64;
    if ( (unsigned int)v50 <= v56 )
    {
LABEL_42:
      for ( i = &v51[2 * v50]; i != v51; v51 += 2 )
        *v51 = -8;
      *(_QWORD *)(a1 + 176) = 0;
      return 0;
    }
    v57 = v51;
    v58 = v49 - 1;
    if ( v58 )
    {
      _BitScanReverse(&v59, v58);
      v60 = 1 << (33 - (v59 ^ 0x1F));
      if ( v60 < 64 )
        v60 = 64;
      if ( (_DWORD)v50 == v60 )
      {
        *(_QWORD *)(a1 + 176) = 0;
        v77 = &v51[2 * v50];
        do
        {
          if ( v57 )
            *v57 = -8;
          v57 += 2;
        }
        while ( v77 != v57 );
        return 0;
      }
      v61 = (4 * v60 / 3u + 1) | ((unsigned __int64)(4 * v60 / 3u + 1) >> 1);
      v62 = ((v61 | (v61 >> 2)) >> 4) | v61 | (v61 >> 2) | ((((v61 | (v61 >> 2)) >> 4) | v61 | (v61 >> 2)) >> 8);
      v63 = (v62 | (v62 >> 16)) + 1;
      v64 = 16 * ((v62 | (v62 >> 16)) + 1);
    }
    else
    {
      v64 = 2048;
      v63 = 128;
    }
    j___libc_free_0(v51);
    *(_DWORD *)(a1 + 184) = v63;
    v65 = (_QWORD *)sub_22077B0(v64);
    v66 = *(unsigned int *)(a1 + 184);
    *(_QWORD *)(a1 + 176) = 0;
    *(_QWORD *)(a1 + 168) = v65;
    for ( j = &v65[2 * v66]; j != v65; v65 += 2 )
    {
      if ( v65 )
        *v65 = -8;
    }
    return 0;
  }
  v13 = v12;
  v14 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9A488, 1u);
  v15 = v14;
  if ( !v14 )
  {
    v69 = sub_13FD000(a2);
    if ( v69 )
    {
      sub_1AFD990(v69, "llvm.loop.unroll.full", 21);
      goto LABEL_7;
    }
    if ( !LOBYTE(qword_50525C0[20]) )
    {
      v18 = a1 + 160;
      sub_143A950(v86, *(__int64 **)(**(_QWORD **)(a2 + 32) + 56LL));
      goto LABEL_10;
    }
LABEL_33:
    v46 = *(__int64 **)(a1 + 8);
    v47 = *v46;
    v48 = v46[1];
    if ( v47 == v48 )
LABEL_124:
      BUG();
    while ( *(_UNKNOWN **)v47 != &unk_4F99768 )
    {
      v47 += 16;
      if ( v48 == v47 )
        goto LABEL_124;
    }
    (*(void (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v47 + 8) + 104LL))(*(_QWORD *)(v47 + 8), &unk_4F99768);
LABEL_8:
    v18 = a1 + 160;
    sub_143A950(v86, *(__int64 **)(**(_QWORD **)(a2 + 32) + 56LL));
    if ( !v15 )
    {
LABEL_10:
      v19 = *(__int64 **)(a1 + 8);
      v20 = *v19;
      v21 = v19[1];
      if ( v20 == v21 )
LABEL_123:
        BUG();
      while ( *(_UNKNOWN **)v20 != &unk_4F9D3C0 )
      {
        v20 += 16;
        if ( v21 == v20 )
          goto LABEL_123;
      }
      v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(
              *(_QWORD *)(v20 + 8),
              &unk_4F9D3C0);
      v23 = sub_14A4050(v22, *(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL));
      v24 = *(__int64 **)(a1 + 8);
      v82 = (__int64 **)v23;
      v25 = *v24;
      v26 = v24[1];
      if ( v25 == v26 )
LABEL_125:
        BUG();
      while ( *(_UNKNOWN **)v25 != &unk_4F9B6E8 )
      {
        v25 += 16;
        if ( v26 == v25 )
          goto LABEL_125;
      }
      v27 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v25 + 8) + 104LL))(
              *(_QWORD *)(v25 + 8),
              &unk_4F9B6E8);
      v28 = *(__int64 **)(a1 + 8);
      v85 = (_QWORD *)(v27 + 360);
      v29 = *v28;
      v30 = v28[1];
      if ( v29 == v30 )
LABEL_126:
        BUG();
      while ( *(_UNKNOWN **)v29 != &unk_4F9E06C )
      {
        v29 += 16;
        if ( v30 == v29 )
          goto LABEL_126;
      }
      v31 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v29 + 8) + 104LL))(
              *(_QWORD *)(v29 + 8),
              &unk_4F9E06C);
      v32 = *(__int64 **)(a1 + 8);
      v84 = v31 + 160;
      v33 = *v32;
      v34 = v32[1];
      if ( v33 == v34 )
LABEL_127:
        BUG();
      while ( *(_UNKNOWN **)v33 != &unk_4F9920C )
      {
        v33 += 16;
        if ( v34 == v33 )
          goto LABEL_127;
      }
      v35 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v33 + 8) + 104LL))(
              *(_QWORD *)(v33 + 8),
              &unk_4F9920C);
      v36 = *(__int64 **)(a1 + 8);
      v37 = v35 + 160;
      v38 = *v36;
      v39 = v36[1];
      if ( v38 == v39 )
LABEL_128:
        BUG();
      while ( *(_UNKNOWN **)v38 != &unk_4F96DB4 )
      {
        v38 += 16;
        if ( v39 == v38 )
          goto LABEL_128;
      }
      v83 = v37;
      v40 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v38 + 8) + 104LL))(
              *(_QWORD *)(v38 + 8),
              &unk_4F96DB4);
      v43 = sub_19665C0(
              v18,
              (_QWORD *)a2,
              *(_QWORD *)(v40 + 160),
              v83,
              v84,
              v85,
              a3,
              a4,
              a5,
              a6,
              v41,
              v42,
              a9,
              a10,
              v82,
              v15,
              (__int64 *)v86,
              0);
      v44 = v87;
      v13 = v43;
      if ( v87 )
      {
        sub_1368A00(v87);
        j_j___libc_free_0(v44, 8);
      }
      return v13;
    }
LABEL_9:
    v15 = *(_QWORD *)(v15 + 160);
    goto LABEL_10;
  }
  v15 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v14 + 104LL))(v14, &unk_4F9A488);
  v16 = sub_13FD000(a2);
  if ( !v16
    || (v17 = sub_1AFD990(v16, "llvm.loop.unroll.full", 21), !v15)
    || !v17
    || !(unsigned int)sub_1474220(*(_QWORD *)(v15 + 160), a2) )
  {
LABEL_7:
    if ( !LOBYTE(qword_50525C0[20]) )
      goto LABEL_8;
    goto LABEL_33;
  }
  v70 = *(_QWORD *)(a2 + 32);
  v71 = *(_QWORD *)(a2 + 40);
  if ( v70 == v71 )
  {
    sub_1474220(*(_QWORD *)(v15 + 160), a2);
    goto LABEL_86;
  }
  v72 = 0;
  do
  {
    v73 = *(_QWORD *)v70 + 40LL;
    v74 = *(_QWORD *)(*(_QWORD *)v70 + 48LL);
    if ( v73 != v74 )
    {
      v75 = 0;
      do
      {
        v74 = *(_QWORD *)(v74 + 8);
        ++v75;
      }
      while ( v73 != v74 );
      v72 += v75;
    }
    v70 += 8;
  }
  while ( v71 != v70 );
  if ( (unsigned int)sub_1474220(*(_QWORD *)(v15 + 160), a2) * v72 <= dword_4FB0460 )
  {
LABEL_86:
    if ( !LOBYTE(qword_50525C0[20]) )
    {
      v18 = a1 + 160;
      sub_143A950(v86, *(__int64 **)(**(_QWORD **)(a2 + 32) + 56LL));
      goto LABEL_9;
    }
    goto LABEL_33;
  }
  v76 = a1 + 160;
  if ( *(_DWORD *)(a1 + 176) )
  {
    v78 = *(_QWORD **)(a1 + 168);
    v79 = &v78[2 * *(unsigned int *)(a1 + 184)];
    if ( v78 != v79 )
    {
      while ( 1 )
      {
        v80 = v78;
        if ( *v78 != -16 && *v78 != -8 )
          break;
        v78 += 2;
        if ( v79 == v78 )
          goto LABEL_92;
      }
      while ( v79 != v80 )
      {
        v81 = v80[1];
        if ( v81 )
        {
          sub_12D5E00(v80[1]);
          j_j___libc_free_0(v81, 72);
        }
        v80 += 2;
        if ( v80 == v79 )
          break;
        while ( *v80 == -8 || *v80 == -16 )
        {
          v80 += 2;
          if ( v79 == v80 )
            goto LABEL_92;
        }
      }
    }
  }
LABEL_92:
  sub_195E9A0(v76);
  return v13;
}
