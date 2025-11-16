// Function: sub_20D75F0
// Address: 0x20d75f0
//
void __fastcall sub_20D75F0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // r12
  char *v7; // rax
  char *v8; // rdx
  __int64 v9; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r8
  int v13; // edx
  __int64 v14; // rcx
  unsigned __int64 v15; // r14
  __int64 i; // r14
  unsigned __int64 v17; // r9
  unsigned __int64 v18; // r12
  __int64 j; // r12
  int v20; // r15d
  __int16 v21; // ax
  __int16 *v22; // rdx
  __int16 v23; // ax
  _QWORD *v24; // r13
  __int16 v25; // ax
  __int64 v26; // rax
  __int64 v27; // rax
  __int16 v28; // dx
  bool v29; // al
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rsi
  _BYTE *v34; // rdx
  char v35; // cl
  _QWORD *v36; // rax
  _QWORD *v37; // rdx
  __int64 v38; // rax
  __int64 m; // r14
  unsigned __int64 v40; // rdx
  __int64 n; // r12
  bool v42; // cf
  __int64 v43; // r15
  __int16 v44; // ax
  __int64 v45; // rsi
  __int64 v46; // rax
  unsigned int ii; // r15d
  char *v48; // rbx
  __int64 v49; // rdi
  __int16 v50; // ax
  __int64 v51; // r14
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int16 v55; // ax
  unsigned __int8 **v56; // r8
  __int64 v57; // rsi
  _QWORD *v58; // rax
  _QWORD *v59; // rdx
  __int64 v60; // rax
  __int64 k; // r14
  _QWORD *v62; // rax
  _QWORD *v63; // rdx
  __int64 v64; // rax
  unsigned __int64 v65; // rax
  char v66; // dl
  __int64 v67; // rsi
  unsigned __int8 *v68; // rsi
  __int64 v69; // rax
  unsigned int v70; // ebx
  void *v71; // r12
  __int64 v72; // rax
  __int64 *v73; // r14
  unsigned int v74; // ebx
  __int64 v75; // r12
  __int64 v76; // r12
  _QWORD *v77; // rax
  __int64 v78; // rbx
  _QWORD *v79; // rbx
  unsigned int *v80; // r13
  __int32 v81; // r15d
  unsigned __int64 v82; // rax
  char *v83; // [rsp+10h] [rbp-110h]
  _QWORD **v84; // [rsp+18h] [rbp-108h]
  _QWORD *v85; // [rsp+20h] [rbp-100h]
  _QWORD **jj; // [rsp+28h] [rbp-F8h]
  __int64 v87; // [rsp+30h] [rbp-F0h]
  unsigned __int8 **v88; // [rsp+30h] [rbp-F0h]
  __int64 v90; // [rsp+38h] [rbp-E8h]
  __int64 v91; // [rsp+40h] [rbp-E0h]
  __int64 v92; // [rsp+40h] [rbp-E0h]
  char *v93; // [rsp+48h] [rbp-D8h]
  int v94; // [rsp+50h] [rbp-D0h]
  unsigned __int64 v95; // [rsp+50h] [rbp-D0h]
  unsigned int *v96; // [rsp+50h] [rbp-D0h]
  int v97; // [rsp+58h] [rbp-C8h]
  _QWORD *v98; // [rsp+58h] [rbp-C8h]
  __int64 v99; // [rsp+68h] [rbp-B8h] BYREF
  __m128i v100; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v101; // [rsp+80h] [rbp-A0h]
  __int64 v102; // [rsp+88h] [rbp-98h]
  __int64 v103; // [rsp+90h] [rbp-90h]
  unsigned __int8 *v104; // [rsp+A0h] [rbp-80h] BYREF
  unsigned int *v105; // [rsp+A8h] [rbp-78h]
  __int64 v106; // [rsp+B0h] [rbp-70h]
  _BYTE v107[32]; // [rsp+B8h] [rbp-68h] BYREF
  unsigned __int64 v108; // [rsp+D8h] [rbp-48h]
  unsigned int v109; // [rsp+E0h] [rbp-40h]

  v2 = a2;
  v3 = *(_QWORD *)(a1 + 112);
  v97 = v2;
  v4 = *(_QWORD *)(a1 + 120);
  v85 = *(_QWORD **)(*(_QWORD *)(v3 + 16 * v2) + 8LL);
  v5 = (v4 - v3) >> 4;
  if ( v4 - v3 < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v6 = 8 * v5;
  if ( v5 )
  {
    v7 = (char *)sub_22077B0(8 * v5);
    v8 = &v7[v6];
    v93 = v7;
    v83 = &v7[v6];
    do
    {
      if ( v7 )
        *(_QWORD *)v7 = 0;
      v7 += 8;
    }
    while ( v7 != v8 );
    v4 = *(_QWORD *)(a1 + 120);
    v3 = *(_QWORD *)(a1 + 112);
  }
  else
  {
    v93 = 0;
    v83 = 0;
  }
  v9 = 0;
  v94 = 0;
  v10 = v85 + 3;
  if ( v3 != v4 )
  {
    while ( 1 )
    {
      if ( v97 == v94 )
        goto LABEL_68;
      v11 = *(_QWORD *)(v3 + 16 * v9 + 8);
      *(_QWORD *)&v93[8 * v9] = v11;
      v12 = *(_QWORD *)(v11 + 24);
      v13 = 0;
      v14 = v12 + 24;
      if ( v11 != v12 + 24 )
        break;
LABEL_16:
      if ( (*(_QWORD *)(v12 + 24) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_166;
      v15 = *(_QWORD *)(v12 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v15 & 4) == 0 && (*(_BYTE *)((*(_QWORD *)(v12 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) != 0 )
      {
        for ( i = *(_QWORD *)(*(_QWORD *)(v12 + 24) & 0xFFFFFFFFFFFFFFF8LL); ; i = *(_QWORD *)v15 )
        {
          v15 = i & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v15 + 46) & 4) == 0 )
            break;
        }
      }
      v91 = v85[3];
      v17 = v91 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v91 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_166;
      v18 = v91 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v17 & 4) == 0 && (*(_BYTE *)(v17 + 46) & 4) != 0 )
      {
        for ( j = *(_QWORD *)v17; ; j = *(_QWORD *)v18 )
        {
          v18 = j & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v18 + 46) & 4) == 0 )
            break;
        }
      }
      v20 = v13 - 1;
      if ( v13 )
      {
        while ( 1 )
        {
          v21 = **(_WORD **)(v15 + 16);
          if ( v21 == 2 || v21 == 12 )
          {
            v58 = (_QWORD *)(*(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL);
            v59 = v58;
            if ( !v58 )
              goto LABEL_166;
            v15 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL;
            v60 = *v58;
            if ( (v60 & 4) != 0 || (*((_BYTE *)v59 + 46) & 4) == 0 )
              goto LABEL_66;
            for ( k = v60; ; k = *(_QWORD *)v15 )
            {
              v15 = k & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v15 + 46) & 4) == 0 )
                break;
            }
            v42 = v20-- == 0;
            if ( v42 )
            {
LABEL_67:
              v3 = *(_QWORD *)(a1 + 112);
              v4 = *(_QWORD *)(a1 + 120);
              break;
            }
          }
          else
          {
            if ( (_QWORD *)v18 == v10 )
            {
              v22 = *(__int16 **)(v18 + 16);
              v24 = (_QWORD *)v18;
              v23 = *v22;
            }
            else
            {
              do
              {
                v22 = *(__int16 **)(v18 + 16);
                v23 = *v22;
                if ( *v22 != 12 && v23 != 2 )
                {
                  v24 = (_QWORD *)v18;
                  goto LABEL_35;
                }
                v62 = (_QWORD *)(*(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL);
                v63 = v62;
                if ( !v62 )
                  goto LABEL_166;
                v18 = *(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL;
                v64 = *v62;
                if ( (v64 & 4) == 0 && (*((_BYTE *)v63 + 46) & 4) != 0 )
                {
                  while ( 1 )
                  {
                    v65 = v64 & 0xFFFFFFFFFFFFFFF8LL;
                    v18 = v65;
                    if ( (*(_BYTE *)(v65 + 46) & 4) == 0 )
                      break;
                    v64 = *(_QWORD *)v65;
                  }
                }
              }
              while ( v10 != (_QWORD *)v18 );
              v22 = (__int16 *)v85[5];
              v24 = v85 + 3;
              v23 = *v22;
            }
LABEL_35:
            if ( v23 == 1 && (*(_BYTE *)(*(_QWORD *)(v18 + 32) + 64LL) & 8) != 0
              || ((v25 = *(_WORD *)(v18 + 46), (v25 & 4) != 0) || (v25 & 8) == 0
                ? (v26 = (*((_QWORD *)v22 + 1) >> 16) & 1LL)
                : (LOBYTE(v26) = sub_1E15D00(v18, 0x10000u, 1)),
                  (_BYTE)v26
               || (v27 = *(_QWORD *)(v18 + 16), *(_WORD *)v27 == 1)
               && (*(_BYTE *)(*(_QWORD *)(v18 + 32) + 64LL) & 0x10) != 0
               || ((v28 = *(_WORD *)(v18 + 46), (v28 & 4) != 0) || (v28 & 8) == 0
                 ? (v29 = (*(_QWORD *)(v27 + 8) & 0x20000LL) != 0)
                 : (v29 = sub_1E15D00(v18, 0x20000u, 1)),
                   v29)) )
            {
              *(_QWORD *)(v18 + 56) = sub_1E15F80(v18, v15);
              *(_BYTE *)(v18 + 49) = v66;
            }
            v30 = *(unsigned int *)(v18 + 40);
            if ( (_DWORD)v30 )
            {
              v31 = 5 * v30;
              v32 = 0;
              v33 = 8 * v31;
              do
              {
                v34 = (_BYTE *)(v32 + *(_QWORD *)(v18 + 32));
                if ( !*v34 )
                {
                  v35 = v34[4];
                  if ( (v35 & 1) != 0 && (*(_BYTE *)(*(_QWORD *)(v15 + 32) + v32 + 4) & 1) == 0 )
                    v34[4] = v35 & 0xFE;
                }
                v32 += 40;
              }
              while ( v33 != v32 );
            }
            v36 = (_QWORD *)(*(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL);
            v37 = v36;
            if ( !v36 )
              goto LABEL_166;
            v15 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL;
            v38 = *v36;
            if ( (v38 & 4) == 0 && (*((_BYTE *)v37 + 46) & 4) != 0 )
            {
              for ( m = v38; ; m = *(_QWORD *)v15 )
              {
                v15 = m & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)(v15 + 46) & 4) == 0 )
                  break;
              }
            }
            v40 = *v24 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v40 )
              goto LABEL_166;
            v18 = *v24 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_QWORD *)v40 & 4) == 0 && (*(_BYTE *)(v40 + 46) & 4) != 0 )
            {
              for ( n = *(_QWORD *)v40; ; n = *(_QWORD *)v18 )
              {
                v18 = n & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)(v18 + 46) & 4) == 0 )
                  break;
              }
            }
LABEL_66:
            v42 = v20-- == 0;
            if ( v42 )
              goto LABEL_67;
          }
        }
      }
LABEL_68:
      v9 = (unsigned int)++v94;
      if ( v94 == (v4 - v3) >> 4 )
        goto LABEL_69;
    }
    while ( 1 )
    {
      ++v13;
      if ( !v11 )
        break;
      if ( (*(_BYTE *)v11 & 4) != 0 )
      {
        v11 = *(_QWORD *)(v11 + 8);
        if ( v14 == v11 )
          goto LABEL_16;
      }
      else
      {
        while ( (*(_BYTE *)(v11 + 46) & 8) != 0 )
          v11 = *(_QWORD *)(v11 + 8);
        v11 = *(_QWORD *)(v11 + 8);
        if ( v14 == v11 )
          goto LABEL_16;
      }
    }
LABEL_166:
    BUG();
  }
LABEL_69:
  v43 = v85[4];
  v95 = (v83 - v93) >> 3;
  while ( v85 + 3 != (_QWORD *)v43 )
  {
    v44 = **(_WORD **)(v43 + 16);
    if ( v44 != 2 && v44 != 12 )
    {
      v45 = *(_QWORD *)(v43 + 64);
      v100.m128i_i64[0] = v45;
      if ( v45 )
        sub_1623A60((__int64)&v100, v45, 2);
      v46 = 0;
      if ( v95 )
      {
        v87 = v43;
        for ( ii = 0; ii < v95; v46 = ++ii )
        {
          if ( v97 != ii )
          {
            v48 = &v93[8 * v46];
            v49 = *(_QWORD *)v48;
            v50 = **(_WORD **)(*(_QWORD *)v48 + 16LL);
            if ( v50 == 2 || v50 == 12 )
            {
              do
              {
                if ( (*(_BYTE *)v49 & 4) == 0 )
                {
                  while ( (*(_BYTE *)(v49 + 46) & 8) != 0 )
                    v49 = *(_QWORD *)(v49 + 8);
                }
                v49 = *(_QWORD *)(v49 + 8);
                *(_QWORD *)v48 = v49;
                v55 = **(_WORD **)(v49 + 16);
              }
              while ( v55 == 12 || v55 == 2 );
            }
            v51 = sub_15C70A0(v49 + 64);
            v52 = sub_15C70A0((__int64)&v100);
            v53 = sub_15BA070(v52, v51, 0);
            sub_15C7080(&v104, v53);
            if ( v100.m128i_i64[0] )
              sub_161E7C0((__int64)&v100, v100.m128i_i64[0]);
            v100.m128i_i64[0] = (__int64)v104;
            if ( v104 )
              sub_1623210((__int64)&v104, v104, (__int64)&v100);
            v54 = *(_QWORD *)v48;
            if ( !*(_QWORD *)v48 )
              goto LABEL_166;
            if ( (*(_BYTE *)v54 & 4) == 0 && (*(_BYTE *)(v54 + 46) & 8) != 0 )
            {
              do
                v54 = *(_QWORD *)(v54 + 8);
              while ( (*(_BYTE *)(v54 + 46) & 8) != 0 );
            }
            *(_QWORD *)v48 = *(_QWORD *)(v54 + 8);
          }
        }
        v43 = v87;
      }
      v56 = (unsigned __int8 **)(v43 + 64);
      v104 = (unsigned __int8 *)v100.m128i_i64[0];
      if ( v100.m128i_i64[0] )
      {
        sub_1623A60((__int64)&v104, v100.m128i_i64[0], 2);
        v56 = (unsigned __int8 **)(v43 + 64);
        if ( (unsigned __int8 **)(v43 + 64) == &v104 )
        {
          if ( v104 )
            sub_161E7C0((__int64)&v104, (__int64)v104);
          goto LABEL_105;
        }
        v67 = *(_QWORD *)(v43 + 64);
        if ( v67 )
        {
LABEL_137:
          v88 = v56;
          sub_161E7C0((__int64)v56, v67);
          v56 = v88;
        }
        v68 = v104;
        *(_QWORD *)(v43 + 64) = v104;
        if ( v68 )
        {
          sub_1623210((__int64)&v104, v68, (__int64)v56);
          v57 = v100.m128i_i64[0];
        }
        else
        {
LABEL_105:
          v57 = v100.m128i_i64[0];
        }
        if ( v57 )
          sub_161E7C0((__int64)&v100, v57);
        goto LABEL_108;
      }
      if ( v56 != &v104 )
      {
        v67 = *(_QWORD *)(v43 + 64);
        if ( v67 )
          goto LABEL_137;
      }
    }
LABEL_108:
    if ( (*(_BYTE *)v43 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v43 + 46) & 8) != 0 )
        v43 = *(_QWORD *)(v43 + 8);
    }
    v43 = *(_QWORD *)(v43 + 8);
  }
  if ( *(_BYTE *)(a1 + 139) )
  {
    v108 = 0;
    v105 = (unsigned int *)v107;
    v69 = *(_QWORD *)(a1 + 160);
    v106 = 0x800000000LL;
    v104 = (unsigned __int8 *)v69;
    v109 = 0;
    v70 = *(_DWORD *)(v69 + 16);
    if ( v70 )
    {
      v71 = _libc_calloc(v70, 1u);
      if ( !v71 )
        sub_16BD1C0("Allocation failed", 1u);
      v108 = (unsigned __int64)v71;
      v109 = v70;
    }
    sub_1DC2B40((__int64)&v104, v85);
    v72 = *(_QWORD *)(a1 + 160);
    v73 = (__int64 *)(a1 + 184);
    *(_DWORD *)(a1 + 200) = 0;
    *(_QWORD *)(a1 + 184) = v72;
    v74 = *(_DWORD *)(v72 + 16);
    if ( v74 < *(_DWORD *)(a1 + 248) >> 2 || v74 > *(_DWORD *)(a1 + 248) )
    {
      _libc_free(*(_QWORD *)(a1 + 240));
      v75 = (__int64)_libc_calloc(v74, 1u);
      if ( !v75 )
      {
        if ( v74 )
          sub_16BD1C0("Allocation failed", 1u);
        else
          v75 = sub_13A3880(1u);
      }
      *(_QWORD *)(a1 + 240) = v75;
      *(_DWORD *)(a1 + 248) = v74;
    }
    v76 = a1;
    v84 = (_QWORD **)v85[9];
    for ( jj = (_QWORD **)v85[8]; v84 != jj; ++jj )
    {
      v77 = *jj;
      *(_DWORD *)(v76 + 200) = 0;
      v78 = (__int64)v77;
      v98 = v77;
      sub_1DC2AE0(v73, v77);
      v79 = (_QWORD *)sub_1DD5EE0(v78);
      v80 = v105;
      v96 = &v105[(unsigned int)v106];
      if ( v96 != v105 )
      {
        do
        {
          v81 = *v80;
          if ( (unsigned __int8)sub_1DC24A0(v73, *(_QWORD *)(v76 + 152), *v80) )
          {
            v99 = 0;
            v90 = v98[7];
            v92 = (__int64)sub_1E0B640(v90, *(_QWORD *)(*(_QWORD *)(v76 + 144) + 8LL) + 576LL, &v99, 0);
            sub_1DD5BA0(v98 + 2, v92);
            v82 = *v79 & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v92 + 8) = v79;
            *(_QWORD *)v92 = v82 | *(_QWORD *)v92 & 7LL;
            *(_QWORD *)(v82 + 8) = v92;
            *v79 = v92 | *v79 & 7LL;
            v100.m128i_i64[0] = 0x10000000;
            v101 = 0;
            v100.m128i_i32[2] = v81;
            v102 = 0;
            v103 = 0;
            sub_1E1A9C0(v92, v90, &v100);
            if ( v99 )
              sub_161E7C0((__int64)&v99, v99);
          }
          ++v80;
        }
        while ( v96 != v80 );
      }
    }
    sub_1DD77B0((__int64)v85);
    sub_1DC3090((__int64)v85, (__int64)&v104);
    _libc_free(v108);
    if ( v105 != (unsigned int *)v107 )
      _libc_free((unsigned __int64)v105);
  }
  if ( v93 )
    j_j___libc_free_0(v93, v83 - v93);
}
