// Function: sub_1365330
// Address: 0x1365330
//
__int64 __fastcall sub_1365330(__int64 *a1, __int64 a2, const __m128i *a3)
{
  __int64 v3; // rax
  unsigned __int8 v4; // al
  __int64 v5; // rdx
  unsigned __int64 v6; // rbx
  __int64 v7; // rcx
  char v8; // r14
  __int64 v9; // rdx
  __int64 *v10; // rbx
  unsigned __int64 v11; // rax
  __int64 *v12; // r15
  unsigned int v13; // r13d
  int v14; // r9d
  char v15; // al
  unsigned int v16; // r9d
  unsigned __int64 v17; // rdi
  char v18; // dl
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // edx
  unsigned int v26; // r12d
  unsigned __int64 v27; // r12
  __int64 v28; // r8
  __int64 v29; // r8
  __int64 v31; // rdi
  __int64 v33; // r15
  __int64 v34; // rbx
  __int64 v35; // rdx
  __int64 v36; // r13
  __int64 v37; // r14
  unsigned int v38; // eax
  unsigned __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rax
  char v43; // al
  _QWORD *v44; // rax
  __m128i *v45; // rdx
  __m128i *v46; // r12
  _BOOL4 v47; // r9d
  __int64 v48; // rax
  _QWORD *v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rax
  unsigned __int64 v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rdx
  int v56; // edx
  __int64 v57; // rax
  __int64 v58; // r13
  char v59; // r14
  __int64 v60; // r13
  char v61; // al
  unsigned __int64 v62; // r13
  __int64 v63; // rdx
  unsigned __int64 v64; // r12
  unsigned __int64 v65; // rbx
  __int64 v66; // rax
  unsigned __int64 v67; // r15
  _QWORD *v68; // r14
  unsigned __int64 v69; // r12
  _QWORD *v70; // rbx
  __int64 v71; // [rsp+0h] [rbp-3C0h]
  __int64 v72; // [rsp+0h] [rbp-3C0h]
  unsigned int v73; // [rsp+8h] [rbp-3B8h]
  unsigned int v74; // [rsp+8h] [rbp-3B8h]
  unsigned __int64 v75; // [rsp+10h] [rbp-3B0h]
  int v76; // [rsp+10h] [rbp-3B0h]
  char v77; // [rsp+10h] [rbp-3B0h]
  unsigned int v78; // [rsp+10h] [rbp-3B0h]
  unsigned __int64 v79; // [rsp+10h] [rbp-3B0h]
  int v80; // [rsp+10h] [rbp-3B0h]
  unsigned __int8 v81; // [rsp+18h] [rbp-3A8h]
  _BOOL4 v82; // [rsp+18h] [rbp-3A8h]
  __int64 v83; // [rsp+18h] [rbp-3A8h]
  unsigned __int64 v85; // [rsp+28h] [rbp-398h]
  unsigned int v86; // [rsp+28h] [rbp-398h]
  int v87; // [rsp+28h] [rbp-398h]
  unsigned __int64 v88; // [rsp+28h] [rbp-398h]
  __int64 v90[2]; // [rsp+38h] [rbp-388h] BYREF
  unsigned __int8 v91; // [rsp+4Fh] [rbp-371h] BYREF
  __m128i v92; // [rsp+50h] [rbp-370h] BYREF
  __int64 (__fastcall *v93)(__m128i **, const __m128i **, int); // [rsp+60h] [rbp-360h]
  __int64 (__fastcall *v94)(); // [rsp+68h] [rbp-358h]
  __int64 v95; // [rsp+70h] [rbp-350h]
  __m128i v96; // [rsp+80h] [rbp-340h] BYREF
  __int64 v97; // [rsp+90h] [rbp-330h] BYREF
  __int64 *v98; // [rsp+98h] [rbp-328h]
  __int64 *v99; // [rsp+A0h] [rbp-320h]
  __int64 v100; // [rsp+A8h] [rbp-318h]

  v90[0] = a2;
  v3 = sub_14AD280(a3->m128i_i64[0], a1[1], 6);
  v85 = sub_1CCAE90(v3, 1);
  v4 = *(_BYTE *)(v85 + 16);
  if ( v4 == 77 )
  {
    v96.m128i_i32[2] = 0;
    v97 = 0;
    v98 = &v96.m128i_i64[1];
    v99 = &v96.m128i_i64[1];
    v100 = 0;
    v92.m128i_i64[0] = v85;
    v44 = sub_1361E90((__int64)&v96, (unsigned __int64 *)&v92);
    v46 = v45;
    if ( v45 )
    {
      v47 = v44 || v45 == (__m128i *)&v96.m128i_u64[1] || v85 < v45[2].m128i_i64[0];
      v82 = v47;
      v48 = sub_22077B0(40);
      *(_QWORD *)(v48 + 32) = v92.m128i_i64[0];
      sub_220F040(v82, v48, v46, &v96.m128i_u64[1]);
      ++v100;
    }
    v91 = 4;
    v93 = 0;
    v49 = (_QWORD *)sub_22077B0(40);
    if ( v49 )
    {
      *v49 = &v92;
      v49[1] = &v91;
      v49[4] = &v96;
      v49[2] = a1;
      v49[3] = v90;
    }
    v92.m128i_i64[0] = (__int64)v49;
    v94 = sub_13660F0;
    v93 = sub_135D900;
    sub_1365EE0(v49, v85, a3->m128i_i64[1]);
    v26 = v91;
    if ( v93 )
      v93((__m128i **)&v92, (const __m128i **)&v92, 3);
    sub_135DB00(v97);
    return v26;
  }
  v5 = v90[0];
  v6 = v90[0] & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_BYTE *)((v90[0] & 0xFFFFFFFFFFFFFFF8LL) + 16) != 78 )
  {
    if ( v4 == 53 )
      goto LABEL_10;
    goto LABEL_9;
  }
  v7 = *(_QWORD *)(v6 - 24);
  if ( *(_BYTE *)(v7 + 16) != 20 || *(_BYTE *)(v7 + 96) )
  {
    if ( v4 != 53 )
      goto LABEL_9;
    goto LABEL_6;
  }
  sub_15F1410(&v96, *(_QWORD *)(v7 + 56), *(_QWORD *)(v7 + 64));
  v33 = v96.m128i_i64[0] + 192LL * v96.m128i_u32[2];
  if ( v96.m128i_i64[0] == v33 )
  {
    if ( (__int64 *)v96.m128i_i64[0] != &v97 )
      _libc_free(v96.m128i_u64[0]);
    return 4;
  }
  v77 = 4;
  v34 = v96.m128i_i64[0];
  while ( !*(_BYTE *)(v34 + 10) )
  {
    if ( (*(_BYTE *)v34 & 2) != 0 )
    {
      v35 = *(_QWORD *)(v34 + 16);
      v36 = v35 + 32LL * *(unsigned int *)(v34 + 24);
      if ( v35 != v36 )
      {
        v37 = *(_QWORD *)(v34 + 16);
        while ( (unsigned int)sub_2241AC0(v37, "{memory}") )
        {
          v37 += 32;
          if ( v36 == v37 )
            goto LABEL_42;
        }
        v77 = 7;
      }
    }
LABEL_42:
    v34 += 192;
    if ( v33 == v34 )
      goto LABEL_100;
  }
  v77 = 7;
LABEL_100:
  v83 = v96.m128i_i64[0];
  v62 = v96.m128i_i64[0] + 192LL * v96.m128i_u32[2];
  if ( v96.m128i_i64[0] != v62 )
  {
    do
    {
      v63 = *(unsigned int *)(v62 - 120);
      v64 = *(_QWORD *)(v62 - 128);
      v62 -= 192LL;
      v65 = v64 + 56 * v63;
      if ( v64 != v65 )
      {
        do
        {
          v66 = *(unsigned int *)(v65 - 40);
          v67 = *(_QWORD *)(v65 - 48);
          v65 -= 56LL;
          v66 *= 32;
          v68 = (_QWORD *)(v67 + v66);
          if ( v67 != v67 + v66 )
          {
            do
            {
              v68 -= 4;
              if ( (_QWORD *)*v68 != v68 + 2 )
                j_j___libc_free_0(*v68, v68[2] + 1LL);
            }
            while ( (_QWORD *)v67 != v68 );
            v67 = *(_QWORD *)(v65 + 8);
          }
          if ( v67 != v65 + 24 )
            _libc_free(v67);
        }
        while ( v64 != v65 );
        v64 = *(_QWORD *)(v62 + 64);
      }
      if ( v64 != v62 + 80 )
        _libc_free(v64);
      v69 = *(_QWORD *)(v62 + 16);
      v70 = (_QWORD *)(v69 + 32LL * *(unsigned int *)(v62 + 24));
      if ( (_QWORD *)v69 != v70 )
      {
        do
        {
          v70 -= 4;
          if ( (_QWORD *)*v70 != v70 + 2 )
            j_j___libc_free_0(*v70, v70[2] + 1LL);
        }
        while ( (_QWORD *)v69 != v70 );
        v69 = *(_QWORD *)(v62 + 16);
      }
      if ( v69 != v62 + 32 )
        _libc_free(v69);
    }
    while ( v83 != v62 );
    v62 = v96.m128i_i64[0];
  }
  if ( (__int64 *)v62 != &v97 )
    _libc_free(v62);
  if ( v77 == 4 )
    return 4;
  v5 = v90[0];
  v4 = *(_BYTE *)(v85 + 16);
  v6 = v90[0] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 == 53 )
  {
    if ( *(_BYTE *)(v6 + 16) != 78 )
      goto LABEL_10;
LABEL_6:
    if ( (*(_WORD *)(v6 + 18) & 3u) - 1 <= 1 )
    {
      v96.m128i_i64[0] = *(_QWORD *)(v6 + 56);
      if ( !(unsigned __int8)sub_1560490(&v96, 6, 0) )
        return 4;
      v5 = v90[0];
      v4 = *(_BYTE *)(v85 + 16);
    }
    v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  }
LABEL_9:
  if ( v4 <= 0x10u )
  {
LABEL_29:
    if ( (unsigned __int8)sub_140B160(v6, a1[3], 0) )
    {
      v96.m128i_i64[0] = v6;
      v96.m128i_i64[1] = -1;
      v97 = 0;
      v31 = *a1;
      v98 = 0;
      v99 = 0;
      if ( !(v31 ? sub_134CB50(v31, (__int64)&v96, (__int64)a3) : (unsigned __int8)sub_1364F20((__int64)a1, &v96, a3)) )
        return 4;
    }
    v27 = v90[0] & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_BYTE *)((v90[0] & 0xFFFFFFFFFFFFFFF8LL) + 16) == 78
      && (v57 = *(_QWORD *)(v27 - 24), !*(_BYTE *)(v57 + 16))
      && (*(_BYTE *)(v57 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v57 + 36) - 133) <= 1
      && v27 )
    {
      v58 = *a1;
      sub_141F670(&v96, v90[0] & 0xFFFFFFFFFFFFFFF8LL);
      if ( v58 )
        v59 = sub_134CB50(v58, (__int64)&v96, (__int64)a3);
      else
        v59 = sub_1364F20((__int64)a1, &v96, a3);
      if ( v59 != 3 )
      {
        v60 = *a1;
        sub_141F750(&v96, v27);
        if ( v60 )
          v61 = sub_134CB50(v60, (__int64)&v96, (__int64)a3);
        else
          v61 = sub_1364F20((__int64)a1, &v96, a3);
        v26 = 6;
        if ( v61 != 3 )
        {
          v26 = (unsigned __int8)(4 - ((v59 == 0) - 1));
          if ( v61 )
            return (unsigned __int8)(4 - ((v59 == 0) - 1)) | 2u;
        }
        return v26;
      }
    }
    else
    {
      if ( sub_135D850(v90[0], 4) )
        return 4;
      if ( !sub_135D850(v28, 79) )
      {
        v26 = 7;
        if ( !sub_135D850(v29, 114) )
          return v26;
      }
    }
    return 5;
  }
LABEL_10:
  if ( v85 == v6 )
    goto LABEL_29;
  v8 = sub_1361F30(a1, v85);
  if ( !v8 )
    goto LABEL_28;
  v9 = 24LL * (*(_DWORD *)((v90[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  v10 = (__int64 *)((v90[0] & 0xFFFFFFFFFFFFFFF8LL) - v9);
  if ( (*(_BYTE *)((v90[0] & 0xFFFFFFFFFFFFFFF8LL) + 23) & 0x40) != 0 )
    v10 = *(__int64 **)((v90[0] & 0xFFFFFFFFFFFFFFF8LL) - 8);
  v11 = (-(__int64)(((v90[0] >> 2) & 1) == 0) & 0xFFFFFFFFFFFFFFD0LL) + v9 - 24;
  v12 = (__int64 *)((char *)v10 + v11);
  if ( (__int64 *)((char *)v10 + v11) == v10 )
    return 4;
  v81 = 4;
  v13 = 0;
  do
  {
    v14 = v13++;
    if ( *(_BYTE *)(*(_QWORD *)*v10 + 8LL) != 15 )
      goto LABEL_16;
    v86 = v14;
    v15 = sub_134FA60(v90, v13, 22);
    v16 = v86;
    if ( v15 )
      goto LABEL_59;
    v17 = v90[0] & 0xFFFFFFFFFFFFFFF8LL;
    v18 = *(_BYTE *)((v90[0] & 0xFFFFFFFFFFFFFFF8LL) + 23);
    v87 = *(_DWORD *)((v90[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF;
    if ( (v90[0] & 4) != 0 )
    {
      if ( v18 < 0 )
      {
        v73 = v16;
        v75 = v90[0] & 0xFFFFFFFFFFFFFFF8LL;
        v19 = sub_1648A40(v17);
        v16 = v73;
        if ( *(char *)(v75 + 23) >= 0 )
        {
          if ( (unsigned int)((v19 + v20) >> 4) )
LABEL_147:
            BUG();
        }
        else
        {
          v71 = v19 + v20;
          v21 = sub_1648A40(v75);
          v16 = v73;
          if ( (unsigned int)((v71 - v21) >> 4) )
          {
            if ( *(char *)(v75 + 23) >= 0 )
              goto LABEL_147;
            v22 = v75;
            v76 = *(_DWORD *)(sub_1648A40(v75) + 8);
            if ( *(char *)(v22 + 23) >= 0 )
              BUG();
            v23 = sub_1648A40(v22);
            v16 = v73;
            v25 = *(_DWORD *)(v23 + v24 - 4) - v76;
            goto LABEL_53;
          }
        }
      }
      v25 = 0;
LABEL_53:
      v38 = v87 - 1 - v25;
      goto LABEL_54;
    }
    if ( v18 < 0 )
    {
      v74 = v16;
      v79 = v90[0] & 0xFFFFFFFFFFFFFFF8LL;
      v50 = sub_1648A40(v17);
      v16 = v74;
      if ( *(char *)(v79 + 23) >= 0 )
      {
        if ( (unsigned int)((v50 + v51) >> 4) )
LABEL_150:
          BUG();
      }
      else
      {
        v72 = v50 + v51;
        v52 = sub_1648A40(v79);
        v16 = v74;
        if ( (unsigned int)((v72 - v52) >> 4) )
        {
          if ( *(char *)(v79 + 23) >= 0 )
            goto LABEL_150;
          v53 = v79;
          v80 = *(_DWORD *)(sub_1648A40(v79) + 8);
          if ( *(char *)(v53 + 23) >= 0 )
            BUG();
          v54 = sub_1648A40(v53);
          v16 = v74;
          v56 = *(_DWORD *)(v54 + v55 - 4) - v80;
          goto LABEL_97;
        }
      }
    }
    v56 = 0;
LABEL_97:
    v38 = v87 - 3 - v56;
LABEL_54:
    if ( v16 < v38 )
    {
      v78 = v16;
      v88 = v90[0] & 0xFFFFFFFFFFFFFFF8LL;
      v39 = (v90[0] & 0xFFFFFFFFFFFFFFF8LL) + 56;
      if ( (v90[0] & 4) != 0 )
      {
        if ( !(unsigned __int8)sub_1560290(v39, v16, 6) )
        {
          v40 = *(_QWORD *)(v88 - 24);
          if ( *(_BYTE *)(v40 + 16) )
            goto LABEL_16;
          goto LABEL_58;
        }
      }
      else if ( !(unsigned __int8)sub_1560290(v39, v16, 6) )
      {
        v40 = *(_QWORD *)(v88 - 72);
        if ( *(_BYTE *)(v40 + 16) )
          goto LABEL_16;
LABEL_58:
        v96.m128i_i64[0] = *(_QWORD *)(v40 + 112);
        if ( !(unsigned __int8)sub_1560290(&v96, v78, 6) )
          goto LABEL_16;
      }
    }
LABEL_59:
    if ( !(unsigned __int8)sub_134FA60(v90, v13, 36) )
    {
      v96.m128i_i64[1] = -1;
      v97 = 0;
      v98 = 0;
      v41 = *a1;
      v99 = 0;
      v96.m128i_i64[0] = a3->m128i_i64[0];
      v42 = *v10;
      v92.m128i_i64[1] = -1;
      v92.m128i_i64[0] = v42;
      v93 = 0;
      v94 = 0;
      v95 = 0;
      v43 = v41 ? sub_134CB50(v41, (__int64)&v92, (__int64)&v96) : sub_1364F20((__int64)a1, &v92, &v96);
      if ( v43 == 3 || (v8 = 0, v43) )
      {
        if ( (unsigned __int8)sub_134FA60(v90, v13, 37) || (unsigned __int8)sub_134FA60(v90, v13, 36) )
        {
          v81 |= 1u;
        }
        else
        {
          if ( !(unsigned __int8)sub_134FA60(v90, v13, 57) && !(unsigned __int8)sub_134FA60(v90, v13, 36) )
            goto LABEL_28;
          v81 |= 2u;
        }
      }
    }
LABEL_16:
    v10 += 3;
  }
  while ( v10 != v12 );
  v26 = v81 & 3;
  if ( (v81 & 3) == 0 )
    return 4;
  if ( (_BYTE)v26 == 3 )
  {
LABEL_28:
    v6 = v90[0] & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_29;
  }
  if ( !v8 )
    return v81 | 4u;
  return v26;
}
