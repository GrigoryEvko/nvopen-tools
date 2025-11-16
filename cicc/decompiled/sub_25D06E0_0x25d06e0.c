// Function: sub_25D06E0
// Address: 0x25d06e0
//
char __fastcall sub_25D06E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  _QWORD *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 *v9; // rdx
  __int64 v10; // r9
  unsigned int v11; // esi
  __int64 *v12; // rcx
  __int64 v13; // r10
  __int64 *v14; // rbx
  _QWORD *v15; // r15
  __int64 *v16; // r13
  __int64 v17; // r12
  size_t v18; // rdx
  __int64 *v19; // rax
  __int64 v20; // r15
  __int64 *v21; // r12
  __int64 v22; // r8
  __int64 v23; // rbx
  unsigned int v24; // r8d
  void *v25; // rsi
  _QWORD *v26; // rdi
  int v27; // eax
  __m128i *v28; // rax
  __m128i v29; // xmm0
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rcx
  int v32; // eax
  __int64 *v33; // rdx
  int v34; // ecx
  int v35; // r8d
  int v36; // eax
  void *v37; // rdx
  const void *v38; // rdi
  __int64 v39; // rcx
  unsigned int v40; // r11d
  int v41; // r8d
  unsigned int v42; // r10d
  __int64 v43; // r9
  const void *v44; // r12
  unsigned int v45; // r10d
  int v46; // eax
  int v47; // eax
  __int64 v49; // [rsp+8h] [rbp-D8h]
  void *v50; // [rsp+10h] [rbp-D0h]
  int v51; // [rsp+24h] [rbp-BCh]
  __int64 v52; // [rsp+28h] [rbp-B8h]
  unsigned int v53; // [rsp+30h] [rbp-B0h]
  unsigned int v54; // [rsp+34h] [rbp-ACh]
  unsigned int v55; // [rsp+34h] [rbp-ACh]
  __int64 v56; // [rsp+38h] [rbp-A8h]
  _QWORD *v59; // [rsp+50h] [rbp-90h]
  bool v60; // [rsp+6Fh] [rbp-71h] BYREF
  void *s1[2]; // [rsp+70h] [rbp-70h] BYREF
  _QWORD v62[12]; // [rsp+80h] [rbp-60h] BYREF

  v3 = *(_QWORD *)(a2 + 40);
  v4 = v3 + 8LL * *(unsigned int *)(a2 + 48);
  v59 = (_QWORD *)v4;
  if ( v4 == v3 )
    return v4;
  v6 = *(_QWORD **)(a2 + 40);
  do
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(a1 + 8);
      v8 = *(_QWORD *)(v7 + 8);
      v4 = *(unsigned int *)(v7 + 24);
      v9 = (__int64 *)(*v6 & 0xFFFFFFFFFFFFFFF8LL);
      v10 = *v9;
      if ( !(_DWORD)v4 )
        goto LABEL_11;
      v11 = (v4 - 1) & (((0xBF58476D1CE4E5B9LL * v10) >> 31) ^ (484763065 * v10));
      v12 = (__int64 *)(v8 + 16LL * v11);
      v13 = *v12;
      if ( v10 != *v12 )
        break;
LABEL_6:
      v4 = v8 + 16 * v4;
      if ( v12 == (__int64 *)v4 )
        goto LABEL_11;
      if ( (unsigned __int64)(v9[4] - v9[3]) > 8 )
      {
        LOBYTE(v4) = *(_BYTE *)(v12[1] + 12) & 0xF;
        switch ( (char)v4 )
        {
          case 0:
          case 1:
          case 3:
          case 5:
          case 6:
          case 7:
          case 8:
            goto LABEL_3;
          case 2:
          case 4:
          case 9:
          case 10:
            LOBYTE(v4) = (*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 16))(*(_QWORD *)(a1 + 24), v10);
            if ( (_BYTE)v4 )
              goto LABEL_3;
            v9 = (__int64 *)(*v6 & 0xFFFFFFFFFFFFFFF8LL);
            break;
          default:
            BUG();
        }
        goto LABEL_11;
      }
LABEL_3:
      if ( v59 == ++v6 )
        return v4;
    }
    v34 = 1;
    while ( v13 != -1 )
    {
      v35 = v34 + 1;
      v11 = (v4 - 1) & (v34 + v11);
      v12 = (__int64 *)(v8 + 16LL * v11);
      v13 = *v12;
      if ( v10 == *v12 )
        goto LABEL_6;
      v34 = v35;
    }
LABEL_11:
    v14 = (__int64 *)v9[4];
    if ( (__int64 *)v9[3] == v14 )
      goto LABEL_3;
    v15 = v6;
    v16 = (__int64 *)v9[3];
    while ( 1 )
    {
      v17 = *v16;
      if ( *(_DWORD *)(*v16 + 8) == 2 )
      {
        v60 = 0;
        if ( *(_QWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 32) - *(_QWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) == 8 )
          break;
        if ( (*(_BYTE *)(v17 + 12) & 0xFu) - 7 > 1 )
          break;
        LOBYTE(v4) = a2;
        v18 = *(_QWORD *)(v17 + 32);
        if ( *(_QWORD *)(a2 + 32) == v18 )
        {
          if ( !v18 )
            break;
          LODWORD(v4) = memcmp(*(const void **)(v17 + 24), *(const void **)(a2 + 24), v18);
          if ( !(_DWORD)v4 )
            break;
        }
      }
LABEL_38:
      if ( v14 == ++v16 )
      {
        v6 = v15;
        goto LABEL_3;
      }
    }
    LOBYTE(v4) = sub_BAF020(*(_QWORD *)a1, v17, 1, &v60);
    if ( !(_BYTE)v4 )
    {
      if ( (_BYTE)qword_4FF0848 && v60 )
        LOBYTE(v4) = sub_25D0520(
                       *(_QWORD *)(a1 + 32),
                       *(_QWORD *)(*v16 + 24),
                       *(_QWORD *)(*v16 + 32),
                       *(_QWORD *)(*v15 & 0xFFFFFFFFFFFFFFF8LL));
      goto LABEL_38;
    }
    v19 = v16;
    v6 = v15;
    v20 = v17;
    v21 = v19;
    LODWORD(v4) = sub_25D0260(
                    *(_QWORD *)(a1 + 32),
                    *(_QWORD *)(*v19 + 24),
                    *(_QWORD *)(*v19 + 32),
                    *(_QWORD *)(*v6 & 0xFFFFFFFFFFFFFFF8LL));
    if ( (_DWORD)v4 != 1 )
      goto LABEL_3;
    v23 = *(_QWORD *)(a1 + 40);
    if ( !v23 )
      goto LABEL_29;
    v24 = *(_DWORD *)(v23 + 24);
    v25 = *(void **)(*v21 + 32);
    v26 = *(_QWORD **)(*v21 + 24);
    s1[1] = v25;
    s1[0] = v26;
    if ( !v24 )
    {
      v62[0] = 0;
      ++*(_QWORD *)v23;
LABEL_24:
      sub_25CE770(v23, 2 * v24);
      goto LABEL_25;
    }
    v54 = v24;
    v56 = *(_QWORD *)(v23 + 8);
    v36 = sub_C94890(v26, (__int64)v25);
    v37 = s1[1];
    v38 = s1[0];
    v39 = 0;
    v40 = v54 - 1;
    v41 = 1;
    v42 = (v54 - 1) & v36;
    while ( 2 )
    {
      v43 = v56 + 48LL * v42;
      v44 = *(const void **)v43;
      if ( *(_QWORD *)v43 != -1 )
      {
        if ( v44 == (const void *)-2LL )
        {
          if ( v38 == (const void *)-2LL )
            goto LABEL_52;
        }
        else
        {
          if ( v37 != *(void **)(v43 + 8) )
          {
LABEL_50:
            v45 = v41 + v42;
            ++v41;
            v42 = v40 & v45;
            continue;
          }
          v51 = v41;
          v52 = v39;
          v53 = v42;
          v55 = v40;
          if ( !v37 )
            goto LABEL_52;
          v49 = v56 + 48LL * v42;
          v50 = v37;
          v46 = memcmp(v38, v44, (size_t)v37);
          v37 = v50;
          v43 = v49;
          v40 = v55;
          v42 = v53;
          v39 = v52;
          v41 = v51;
          if ( !v46 )
            goto LABEL_52;
        }
        if ( !v39 && v44 == (const void *)-2LL )
          v39 = v43;
        goto LABEL_50;
      }
      break;
    }
    if ( v38 == (const void *)-1LL )
    {
LABEL_52:
      sub_25CF280((__int64)v62, v43 + 16, v6);
      v4 = *(_QWORD *)a1;
      if ( !*(_BYTE *)(*(_QWORD *)a1 + 337LL) )
        goto LABEL_30;
      goto LABEL_44;
    }
    v24 = *(_DWORD *)(v23 + 24);
    if ( !v39 )
      v39 = v56 + 48LL * v42;
    v62[0] = v39;
    v47 = *(_DWORD *)(v23 + 16);
    ++*(_QWORD *)v23;
    v27 = v47 + 1;
    if ( 4 * v27 >= 3 * v24 )
      goto LABEL_24;
    if ( v24 - (v27 + *(_DWORD *)(v23 + 20)) > v24 >> 3 )
      goto LABEL_26;
    sub_25CE770(v23, v24);
LABEL_25:
    sub_25CE0A0(v23, (__int64)s1, v62);
    v27 = *(_DWORD *)(v23 + 16) + 1;
LABEL_26:
    *(_DWORD *)(v23 + 16) = v27;
    v28 = (__m128i *)v62[0];
    if ( *(_QWORD *)v62[0] != -1 )
      --*(_DWORD *)(v23 + 20);
    v29 = _mm_loadu_si128((const __m128i *)s1);
    v28[1].m128i_i64[0] = 0;
    v28[1].m128i_i64[1] = 0;
    v28[2].m128i_i64[0] = 0;
    v28[2].m128i_i32[2] = 0;
    *v28 = v29;
    sub_25CF280((__int64)v62, (__int64)v28[1].m128i_i64, v6);
LABEL_29:
    v4 = *(_QWORD *)a1;
    if ( *(_BYTE *)(*(_QWORD *)a1 + 337LL) )
    {
LABEL_44:
      if ( (*(_BYTE *)(v20 + 64) & 2) == 0 )
        goto LABEL_30;
      goto LABEL_3;
    }
LABEL_30:
    v30 = *(unsigned int *)(a3 + 8);
    v31 = *(unsigned int *)(a3 + 12);
    v32 = *(_DWORD *)(a3 + 8);
    if ( v30 >= v31 )
    {
      if ( v31 < v30 + 1 )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v30 + 1, 8u, v22, v30 + 1);
        v30 = *(unsigned int *)(a3 + 8);
      }
      v4 = *(_QWORD *)a3;
      *(_QWORD *)(*(_QWORD *)a3 + 8 * v30) = v20;
      ++*(_DWORD *)(a3 + 8);
      goto LABEL_3;
    }
    v33 = (__int64 *)(*(_QWORD *)a3 + 8 * v30);
    if ( v33 )
    {
      *v33 = v20;
      v32 = *(_DWORD *)(a3 + 8);
    }
    LODWORD(v4) = v32 + 1;
    ++v6;
    *(_DWORD *)(a3 + 8) = v4;
  }
  while ( v59 != v6 );
  return v4;
}
