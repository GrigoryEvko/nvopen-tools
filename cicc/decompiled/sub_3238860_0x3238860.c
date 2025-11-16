// Function: sub_3238860
// Address: 0x3238860
//
__int64 __fastcall sub_3238860(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r13
  int v11; // edx
  __int64 v12; // rax
  unsigned __int8 v13; // al
  __int64 *v14; // rdx
  __int64 v15; // rdx
  const char *v16; // rdi
  unsigned __int8 v17; // dl
  __int64 v18; // rax
  __int64 v19; // rcx
  int v20; // r15d
  __int64 v21; // rax
  unsigned __int64 v22; // r14
  __int64 *v23; // r14
  __int64 v24; // rax
  __int64 (*v25)(); // rdx
  char v26; // si
  __int64 v27; // rdi
  unsigned int v28; // r15d
  unsigned __int8 v29; // al
  __int64 v30; // rcx
  bool v31; // dl
  __int64 *v32; // r8
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int8 v38; // al
  __int64 *v39; // rdx
  unsigned __int8 v40; // dl
  __int64 *v41; // rax
  const char *v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // r8
  __m128i v45; // xmm0
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // r12
  unsigned int v52; // esi
  __int64 v53; // r8
  unsigned int v54; // edi
  __int64 *v55; // rax
  __int64 v56; // rcx
  int v57; // r11d
  __int64 *v58; // rdx
  int v59; // eax
  int v60; // ecx
  __int64 *v61; // rcx
  int v62; // r9d
  int v63; // eax
  int v64; // esi
  __int64 v65; // r8
  unsigned int v66; // eax
  __int64 v67; // rdi
  int v68; // r10d
  __int64 *v69; // r9
  int v70; // eax
  int v71; // eax
  __int64 v72; // rdi
  __int64 *v73; // r8
  unsigned int v74; // r14d
  int v75; // r9d
  __int64 v76; // rsi
  int v77; // [rsp+38h] [rbp-88h]
  void (__fastcall *v78)(__int64 *, _QWORD, _QWORD, const char *, __int64, _QWORD, __int64, __int64, char, __int64, __int64, __int64); // [rsp+38h] [rbp-88h]
  __int64 v79; // [rsp+40h] [rbp-80h] BYREF
  __int64 v80; // [rsp+48h] [rbp-78h] BYREF
  __m128i v81; // [rsp+50h] [rbp-70h] BYREF
  char v82; // [rsp+60h] [rbp-60h]
  __int64 v83; // [rsp+70h] [rbp-50h] BYREF
  __int64 v84; // [rsp+78h] [rbp-48h]
  __int64 v85; // [rsp+80h] [rbp-40h]

  v4 = *(unsigned int *)(a1 + 648);
  v5 = *(_QWORD *)(a1 + 632);
  if ( (_DWORD)v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( *v7 == a2 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
      {
        v9 = *(_QWORD *)(*(_QWORD *)(a1 + 656) + 16LL * *((unsigned int *)v7 + 2) + 8);
        if ( v9 )
          return v9;
      }
    }
    else
    {
      v11 = 1;
      while ( v8 != -4096 )
      {
        v62 = v11 + 1;
        v6 = (v4 - 1) & (v11 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( *v7 == a2 )
          goto LABEL_3;
        v11 = v62;
      }
    }
  }
  if ( *(_BYTE *)(a1 + 3769)
    && !(unsigned __int8)sub_321F6A0(a1, v5)
    && (!*(_BYTE *)(a2 + 41) || *(_DWORD *)(a2 + 32) == 1)
    && *(_DWORD *)(a1 + 664) )
  {
    return *(_QWORD *)(*(_QWORD *)(a1 + 656) + 8LL);
  }
  v12 = a2;
  if ( *(_BYTE *)a2 == 16
    || ((v13 = *(_BYTE *)(a2 - 16), (v13 & 2) != 0)
      ? (v14 = *(__int64 **)(a2 - 32))
      : (v14 = (__int64 *)(a2 - 16 - 8LL * ((v13 >> 2) & 0xF))),
        (v12 = *v14) != 0) )
  {
    v17 = *(_BYTE *)(v12 - 16);
    if ( (v17 & 2) != 0 )
      v18 = *(_QWORD *)(v12 - 32);
    else
      v18 = v12 - 16 - 8LL * ((v17 >> 2) & 0xF);
    v16 = *(const char **)(v18 + 8);
    if ( v16 )
      v16 = (const char *)sub_B91420((__int64)v16);
    else
      v15 = 0;
  }
  else
  {
    v15 = 0;
    v16 = byte_3F871B3;
  }
  *(_QWORD *)(a1 + 3064) = v16;
  v19 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 3072) = v15;
  v20 = *(_DWORD *)(a1 + 3240);
  v77 = v19;
  v21 = sub_22077B0(0x310u);
  v9 = v21;
  if ( v21 )
    sub_37358C0(v21, v20, a2, v77, a1, a1 + 3080, 1);
  v83 = v9;
  sub_3245240(a1 + 3080, &v83);
  v22 = v83;
  if ( v83 )
  {
    sub_3223CF0(v83);
    j_j___libc_free_0(v22);
  }
  v23 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 224LL);
  v24 = *v23;
  v25 = *(__int64 (**)())(*v23 + 104);
  if ( v25 != sub_C13EF0 )
  {
    if ( ((unsigned __int8 (__fastcall *)(_QWORD))v25)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL))
      && !*(_BYTE *)(a1 + 4800) )
    {
      goto LABEL_42;
    }
    v23 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 224LL);
    v24 = *v23;
  }
  v26 = *(_BYTE *)a2;
  v27 = a2;
  v28 = *(_DWORD *)(v9 + 72);
  v78 = *(void (__fastcall **)(__int64 *, _QWORD, _QWORD, const char *, __int64, _QWORD, __int64, __int64, char, __int64, __int64, __int64))(v24 + 664);
  if ( *(_BYTE *)a2 == 16
    || ((v29 = *(_BYTE *)(a2 - 16), v30 = a2 - 16, v31 = (v29 & 2) != 0, (v29 & 2) == 0)
      ? (v32 = (__int64 *)(v30 - 8LL * ((v29 >> 2) & 0xF)))
      : (v32 = *(__int64 **)(a2 - 32)),
        (v27 = *v32) != 0) )
  {
    v33 = *(_QWORD *)(v27 + 40);
    if ( v33 )
    {
      v34 = sub_B91420(v33);
      LOBYTE(v85) = 1;
      v26 = *(_BYTE *)a2;
      v83 = v34;
      v84 = v35;
    }
    else
    {
      LOBYTE(v85) = 0;
    }
    v36 = a2;
    if ( v26 == 16 )
      goto LABEL_33;
    v29 = *(_BYTE *)(a2 - 16);
    v30 = a2 - 16;
    v31 = (v29 & 2) != 0;
  }
  else
  {
    LOBYTE(v85) = 0;
  }
  if ( v31 )
    v61 = *(__int64 **)(a2 - 32);
  else
    v61 = (__int64 *)(v30 - 8LL * ((v29 >> 2) & 0xF));
  v36 = *v61;
LABEL_33:
  sub_3222AF0(&v81, a1, v36);
  v37 = a2;
  if ( *(_BYTE *)a2 == 16
    || ((v38 = *(_BYTE *)(a2 - 16), (v38 & 2) == 0)
      ? (v39 = (__int64 *)(a2 - 16 - 8LL * ((v38 >> 2) & 0xF)))
      : (v39 = *(__int64 **)(a2 - 32)),
        (v37 = *v39) != 0) )
  {
    v40 = *(_BYTE *)(v37 - 16);
    if ( (v40 & 2) != 0 )
      v41 = *(__int64 **)(v37 - 32);
    else
      v41 = (__int64 *)(v37 - 16 - 8LL * ((v40 >> 2) & 0xF));
    v42 = (const char *)*v41;
    if ( *v41 )
    {
      v42 = (const char *)sub_B91420(*v41);
      v44 = v43;
    }
    else
    {
      v44 = 0;
    }
  }
  else
  {
    v44 = 0;
    v42 = byte_3F871B3;
  }
  v45 = _mm_loadu_si128(&v81);
  v78(
    v23,
    *(_QWORD *)(a1 + 3064),
    *(_QWORD *)(a1 + 3072),
    v42,
    v44,
    v28,
    v45.m128i_i64[0],
    v45.m128i_i64[1],
    v82,
    v83,
    v84,
    v85);
LABEL_42:
  if ( *(_BYTE *)(a1 + 3769) )
  {
    v46 = sub_3224A90(a1, v9);
    v47 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(v9 + 408) = v46;
    *(_QWORD *)(v9 + 56) = *(_QWORD *)(sub_31DA6B0(v47) + 232);
  }
  else
  {
    sub_3221260(a1, a2, v9);
    *(_QWORD *)(v9 + 56) = *(_QWORD *)(sub_31DA6B0(*(_QWORD *)(a1 + 8)) + 88);
  }
  v79 = a2;
  v80 = v9;
  v51 = v9 + 8;
  sub_3238550(a1 + 624, &v79, &v80, v48, v49, v50);
  v52 = *(_DWORD *)(a1 + 696);
  if ( !v52 )
  {
    ++*(_QWORD *)(a1 + 672);
    goto LABEL_79;
  }
  v53 = *(_QWORD *)(a1 + 680);
  v54 = (v52 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
  v55 = (__int64 *)(v53 + 16LL * v54);
  v56 = *v55;
  if ( v51 == *v55 )
    return v9;
  v57 = 1;
  v58 = 0;
  while ( v56 != -4096 )
  {
    if ( v56 != -8192 || v58 )
      v55 = v58;
    v54 = (v52 - 1) & (v57 + v54);
    v56 = *(_QWORD *)(v53 + 16LL * v54);
    if ( v51 == v56 )
      return v9;
    ++v57;
    v58 = v55;
    v55 = (__int64 *)(v53 + 16LL * v54);
  }
  if ( !v58 )
    v58 = v55;
  v59 = *(_DWORD *)(a1 + 688);
  ++*(_QWORD *)(a1 + 672);
  v60 = v59 + 1;
  if ( 4 * (v59 + 1) >= 3 * v52 )
  {
LABEL_79:
    sub_3229060(a1 + 672, 2 * v52);
    v63 = *(_DWORD *)(a1 + 696);
    if ( v63 )
    {
      v64 = v63 - 1;
      v65 = *(_QWORD *)(a1 + 680);
      v66 = (v63 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v60 = *(_DWORD *)(a1 + 688) + 1;
      v58 = (__int64 *)(v65 + 16LL * v66);
      v67 = *v58;
      if ( v51 != *v58 )
      {
        v68 = 1;
        v69 = 0;
        while ( v67 != -4096 )
        {
          if ( v67 == -8192 && !v69 )
            v69 = v58;
          v66 = v64 & (v68 + v66);
          v58 = (__int64 *)(v65 + 16LL * v66);
          v67 = *v58;
          if ( v51 == *v58 )
            goto LABEL_52;
          ++v68;
        }
        if ( v69 )
          v58 = v69;
      }
      goto LABEL_52;
    }
    goto LABEL_107;
  }
  if ( v52 - *(_DWORD *)(a1 + 692) - v60 <= v52 >> 3 )
  {
    sub_3229060(a1 + 672, v52);
    v70 = *(_DWORD *)(a1 + 696);
    if ( v70 )
    {
      v71 = v70 - 1;
      v72 = *(_QWORD *)(a1 + 680);
      v73 = 0;
      v74 = v71 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v75 = 1;
      v60 = *(_DWORD *)(a1 + 688) + 1;
      v58 = (__int64 *)(v72 + 16LL * v74);
      v76 = *v58;
      if ( v51 != *v58 )
      {
        while ( v76 != -4096 )
        {
          if ( v76 == -8192 && !v73 )
            v73 = v58;
          v74 = v71 & (v75 + v74);
          v58 = (__int64 *)(v72 + 16LL * v74);
          v76 = *v58;
          if ( v51 == *v58 )
            goto LABEL_52;
          ++v75;
        }
        if ( v73 )
          v58 = v73;
      }
      goto LABEL_52;
    }
LABEL_107:
    ++*(_DWORD *)(a1 + 688);
    BUG();
  }
LABEL_52:
  *(_DWORD *)(a1 + 688) = v60;
  if ( *v58 != -4096 )
    --*(_DWORD *)(a1 + 692);
  *v58 = v51;
  v58[1] = v9;
  return v9;
}
