// Function: sub_18A4390
// Address: 0x18a4390
//
__int64 __fastcall sub_18A4390(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  _QWORD *v4; // r13
  _QWORD *v5; // r14
  __int64 v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r13
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // rbx
  _QWORD *v14; // r13
  unsigned __int64 v15; // rdi
  void (__fastcall *v16)(__int64, __int64, __int64); // rax
  void (__fastcall *v17)(__int64, __int64, __int64); // rax
  __int64 v18; // r14
  int v19; // eax
  unsigned int v20; // ecx
  __int64 v21; // rdx
  _QWORD *v22; // rax
  _QWORD *j; // rdx
  __int64 v24; // r15
  __int64 *v25; // rbx
  __int64 *v26; // r13
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rax
  void *v30; // rdi
  unsigned int v31; // eax
  __int64 v32; // rdx
  unsigned __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rdi
  __int64 v37; // rax
  unsigned __int64 *v38; // rbx
  unsigned __int64 *v39; // r13
  unsigned __int64 v40; // rdi
  __int64 v41; // rax
  unsigned __int64 *v42; // rdx
  unsigned __int64 v43; // rcx
  unsigned __int64 *v44; // r13
  unsigned __int64 *v45; // rbx
  unsigned __int64 v46; // rdi
  unsigned __int64 *v47; // rbx
  unsigned __int64 v48; // r13
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // rdi
  __int64 v51; // rdi
  __int64 v52; // r15
  __int64 v53; // rax
  _QWORD *v54; // rbx
  _QWORD *v55; // r13
  __int64 v56; // r14
  __int64 v57; // rdi
  __int64 v58; // r15
  __int64 v59; // rax
  _QWORD *v60; // rbx
  _QWORD *v61; // r13
  __int64 v62; // r14
  __int64 v63; // rdi
  unsigned __int64 v64; // r8
  __int64 v65; // rax
  __int64 v66; // r13
  __int64 v67; // rbx
  unsigned __int64 v68; // rdi
  unsigned __int64 v69; // rdi
  unsigned __int64 v70; // rdi
  _QWORD *v72; // rdi
  unsigned int v73; // eax
  int v74; // r13d
  unsigned int v75; // eax
  _QWORD *v76; // rax
  __int64 v77; // rdx
  _QWORD *i; // rdx
  _QWORD *v79; // rax
  __int64 *v80; // [rsp+0h] [rbp-40h]
  __int64 *v81; // [rsp+8h] [rbp-38h]

  sub_2240A30(a1 + 1208);
  v2 = *(_QWORD *)(a1 + 1192);
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = *(unsigned int *)(a1 + 1176);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 1160);
    v5 = &v4[7 * v3];
    do
    {
      if ( *v4 != -8 && *v4 != -16 )
      {
        v6 = v4[3];
        while ( v6 )
        {
          sub_18A3F70(*(_QWORD *)(v6 + 24));
          v7 = v6;
          v6 = *(_QWORD *)(v6 + 16);
          j_j___libc_free_0(v7, 48);
        }
      }
      v4 += 7;
    }
    while ( v5 != v4 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1160));
  v8 = *(unsigned int *)(a1 + 1144);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD **)(a1 + 1128);
    v10 = &v9[11 * v8];
    do
    {
      if ( *v9 != -16 && *v9 != -8 )
      {
        v11 = v9[1];
        if ( (_QWORD *)v11 != v9 + 3 )
          _libc_free(v11);
      }
      v9 += 11;
    }
    while ( v10 != v9 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1128));
  v12 = *(unsigned int *)(a1 + 1112);
  if ( (_DWORD)v12 )
  {
    v13 = *(_QWORD **)(a1 + 1096);
    v14 = &v13[11 * v12];
    do
    {
      if ( *v13 != -16 && *v13 != -8 )
      {
        v15 = v13[1];
        if ( (_QWORD *)v15 != v13 + 3 )
          _libc_free(v15);
      }
      v13 += 11;
    }
    while ( v14 != v13 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 1096));
  v16 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 1072);
  if ( v16 )
    v16(a1 + 1056, a1 + 1056, 3);
  v17 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 1040);
  if ( v17 )
    v17(a1 + 1024, a1 + 1024, 3);
  v18 = *(_QWORD *)(a1 + 1016);
  if ( v18 )
  {
    v19 = *(_DWORD *)(v18 + 16);
    ++*(_QWORD *)v18;
    if ( v19 )
    {
      v20 = 4 * v19;
      v21 = *(unsigned int *)(v18 + 24);
      if ( (unsigned int)(4 * v19) < 0x40 )
        v20 = 64;
      if ( (unsigned int)v21 <= v20 )
        goto LABEL_33;
      v72 = *(_QWORD **)(v18 + 8);
      v73 = v19 - 1;
      if ( v73 )
      {
        _BitScanReverse(&v73, v73);
        v74 = 1 << (33 - (v73 ^ 0x1F));
        if ( v74 < 64 )
          v74 = 64;
        if ( (_DWORD)v21 == v74 )
        {
          *(_QWORD *)(v18 + 16) = 0;
          v79 = &v72[2 * (unsigned int)v21];
          do
          {
            if ( v72 )
              *v72 = -8;
            v72 += 2;
          }
          while ( v79 != v72 );
LABEL_36:
          v80 = *(__int64 **)(v18 + 40);
          if ( *(__int64 **)(v18 + 32) != v80 )
          {
            v81 = *(__int64 **)(v18 + 32);
            do
            {
              v24 = *v81;
              v25 = *(__int64 **)(*v81 + 16);
              if ( *(__int64 **)(*v81 + 8) == v25 )
              {
                *(_BYTE *)(v24 + 160) = 1;
              }
              else
              {
                v26 = *(__int64 **)(*v81 + 8);
                do
                {
                  v27 = *v26++;
                  sub_13FACC0(v27);
                }
                while ( v25 != v26 );
                *(_BYTE *)(v24 + 160) = 1;
                v28 = *(_QWORD *)(v24 + 8);
                if ( *(_QWORD *)(v24 + 16) != v28 )
                  *(_QWORD *)(v24 + 16) = v28;
              }
              v29 = *(_QWORD *)(v24 + 32);
              if ( v29 != *(_QWORD *)(v24 + 40) )
                *(_QWORD *)(v24 + 40) = v29;
              ++*(_QWORD *)(v24 + 56);
              v30 = *(void **)(v24 + 72);
              if ( v30 == *(void **)(v24 + 64) )
              {
                *(_QWORD *)v24 = 0;
              }
              else
              {
                v31 = 4 * (*(_DWORD *)(v24 + 84) - *(_DWORD *)(v24 + 88));
                v32 = *(unsigned int *)(v24 + 80);
                if ( v31 < 0x20 )
                  v31 = 32;
                if ( (unsigned int)v32 > v31 )
                  sub_16CC920(v24 + 56);
                else
                  memset(v30, -1, 8 * v32);
                v33 = *(_QWORD *)(v24 + 72);
                v34 = *(_QWORD *)(v24 + 64);
                *(_QWORD *)v24 = 0;
                if ( v33 != v34 )
                  _libc_free(v33);
              }
              v35 = *(_QWORD *)(v24 + 32);
              if ( v35 )
                j_j___libc_free_0(v35, *(_QWORD *)(v24 + 48) - v35);
              v36 = *(_QWORD *)(v24 + 8);
              if ( v36 )
                j_j___libc_free_0(v36, *(_QWORD *)(v24 + 24) - v36);
              ++v81;
            }
            while ( v80 != v81 );
            v37 = *(_QWORD *)(v18 + 32);
            if ( *(_QWORD *)(v18 + 40) != v37 )
              *(_QWORD *)(v18 + 40) = v37;
          }
          v38 = *(unsigned __int64 **)(v18 + 120);
          v39 = &v38[2 * *(unsigned int *)(v18 + 128)];
          while ( v39 != v38 )
          {
            v40 = *v38;
            v38 += 2;
            _libc_free(v40);
          }
          *(_DWORD *)(v18 + 128) = 0;
          v41 = *(unsigned int *)(v18 + 80);
          if ( (_DWORD)v41 )
          {
            *(_QWORD *)(v18 + 136) = 0;
            v42 = *(unsigned __int64 **)(v18 + 72);
            v43 = *v42;
            v44 = &v42[v41];
            v45 = v42 + 1;
            *(_QWORD *)(v18 + 56) = *v42;
            *(_QWORD *)(v18 + 64) = v43 + 4096;
            if ( v44 != v42 + 1 )
            {
              do
              {
                v46 = *v45++;
                _libc_free(v46);
              }
              while ( v44 != v45 );
              v42 = *(unsigned __int64 **)(v18 + 72);
            }
            *(_DWORD *)(v18 + 80) = 1;
            _libc_free(*v42);
            v47 = *(unsigned __int64 **)(v18 + 120);
            v48 = (unsigned __int64)&v47[2 * *(unsigned int *)(v18 + 128)];
            if ( v47 == (unsigned __int64 *)v48 )
              goto LABEL_68;
            do
            {
              v49 = *v47;
              v47 += 2;
              _libc_free(v49);
            }
            while ( v47 != (unsigned __int64 *)v48 );
          }
          v48 = *(_QWORD *)(v18 + 120);
LABEL_68:
          if ( v48 != v18 + 136 )
            _libc_free(v48);
          v50 = *(_QWORD *)(v18 + 72);
          if ( v50 != v18 + 88 )
            _libc_free(v50);
          v51 = *(_QWORD *)(v18 + 32);
          if ( v51 )
            j_j___libc_free_0(v51, *(_QWORD *)(v18 + 48) - v51);
          j___libc_free_0(*(_QWORD *)(v18 + 8));
          j_j___libc_free_0(v18, 160);
          goto LABEL_75;
        }
      }
      else
      {
        v74 = 64;
      }
      j___libc_free_0(v72);
      v75 = sub_18A4140(v74);
      *(_DWORD *)(v18 + 24) = v75;
      if ( v75 )
      {
        v76 = (_QWORD *)sub_22077B0(16LL * v75);
        v77 = *(unsigned int *)(v18 + 24);
        *(_QWORD *)(v18 + 16) = 0;
        *(_QWORD *)(v18 + 8) = v76;
        for ( i = &v76[2 * v77]; i != v76; v76 += 2 )
        {
          if ( v76 )
            *v76 = -8;
        }
        goto LABEL_36;
      }
    }
    else
    {
      if ( !*(_DWORD *)(v18 + 20) )
        goto LABEL_36;
      v21 = *(unsigned int *)(v18 + 24);
      if ( (unsigned int)v21 <= 0x40 )
      {
LABEL_33:
        v22 = *(_QWORD **)(v18 + 8);
        for ( j = &v22[2 * v21]; j != v22; v22 += 2 )
          *v22 = -8;
        goto LABEL_35;
      }
      j___libc_free_0(*(_QWORD *)(v18 + 8));
      *(_DWORD *)(v18 + 24) = 0;
    }
    *(_QWORD *)(v18 + 8) = 0;
LABEL_35:
    *(_QWORD *)(v18 + 16) = 0;
    goto LABEL_36;
  }
LABEL_75:
  v52 = *(_QWORD *)(a1 + 1008);
  if ( v52 )
  {
    v53 = *(unsigned int *)(v52 + 72);
    if ( (_DWORD)v53 )
    {
      v54 = *(_QWORD **)(v52 + 56);
      v55 = &v54[2 * v53];
      do
      {
        if ( *v54 != -16 && *v54 != -8 )
        {
          v56 = v54[1];
          if ( v56 )
          {
            v57 = *(_QWORD *)(v56 + 24);
            if ( v57 )
              j_j___libc_free_0(v57, *(_QWORD *)(v56 + 40) - v57);
            j_j___libc_free_0(v56, 56);
          }
        }
        v54 += 2;
      }
      while ( v55 != v54 );
    }
    j___libc_free_0(*(_QWORD *)(v52 + 56));
    if ( *(_QWORD *)v52 != v52 + 16 )
      _libc_free(*(_QWORD *)v52);
    j_j___libc_free_0(v52, 104);
  }
  v58 = *(_QWORD *)(a1 + 1000);
  if ( v58 )
  {
    v59 = *(unsigned int *)(v58 + 48);
    if ( (_DWORD)v59 )
    {
      v60 = *(_QWORD **)(v58 + 32);
      v61 = &v60[2 * v59];
      do
      {
        if ( *v60 != -8 && *v60 != -16 )
        {
          v62 = v60[1];
          if ( v62 )
          {
            v63 = *(_QWORD *)(v62 + 24);
            if ( v63 )
              j_j___libc_free_0(v63, *(_QWORD *)(v62 + 40) - v63);
            j_j___libc_free_0(v62, 56);
          }
        }
        v60 += 2;
      }
      while ( v61 != v60 );
    }
    j___libc_free_0(*(_QWORD *)(v58 + 32));
    if ( *(_QWORD *)v58 != v58 + 16 )
      _libc_free(*(_QWORD *)v58);
    j_j___libc_free_0(v58, 80);
  }
  v64 = *(_QWORD *)(a1 + 968);
  if ( *(_DWORD *)(a1 + 980) )
  {
    v65 = *(unsigned int *)(a1 + 976);
    if ( (_DWORD)v65 )
    {
      v66 = 8 * v65;
      v67 = 0;
      do
      {
        v68 = *(_QWORD *)(v64 + v67);
        if ( v68 != -8 && v68 )
        {
          _libc_free(v68);
          v64 = *(_QWORD *)(a1 + 968);
        }
        v67 += 8;
      }
      while ( v67 != v66 );
    }
  }
  _libc_free(v64);
  j___libc_free_0(*(_QWORD *)(a1 + 944));
  sub_18A3DA0(*(_QWORD *)(a1 + 904));
  v69 = *(_QWORD *)(a1 + 360);
  if ( v69 != a1 + 376 )
    _libc_free(v69);
  v70 = *(_QWORD *)(a1 + 80);
  if ( v70 != *(_QWORD *)(a1 + 72) )
    _libc_free(v70);
  j___libc_free_0(*(_QWORD *)(a1 + 40));
  return j___libc_free_0(*(_QWORD *)(a1 + 8));
}
