// Function: sub_26C0C70
// Address: 0x26c0c70
//
__int64 __fastcall sub_26C0C70(__int64 a1)
{
  volatile signed __int32 *v2; // rdi
  unsigned __int64 v3; // r13
  __int64 v4; // rdi
  __int64 v5; // rax
  _QWORD *v6; // r13
  _QWORD *v7; // r14
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *v16; // r13
  unsigned __int64 v17; // rdi
  __int64 v18; // rsi
  unsigned __int64 v19; // r14
  int v20; // eax
  unsigned int v21; // ecx
  __int64 v22; // rdx
  _QWORD *v23; // rax
  _QWORD *j; // rdx
  __int64 v25; // r15
  __int64 *v26; // rbx
  __int64 *v27; // r13
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned int v31; // eax
  __int64 v32; // rdx
  char v33; // al
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  __int64 v36; // rax
  __int64 *v37; // rbx
  __int64 *v38; // r13
  __int64 v39; // rsi
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 *v42; // rax
  __int64 v43; // rcx
  __int64 *v44; // rbx
  __int64 *v45; // r15
  __int64 v46; // rdi
  unsigned int v47; // ecx
  __int64 v48; // rsi
  __int64 *v49; // rbx
  unsigned __int64 v50; // r13
  __int64 v51; // rsi
  __int64 v52; // rdi
  unsigned __int64 v53; // rdi
  unsigned __int64 v54; // rdi
  unsigned __int64 v55; // r15
  __int64 v56; // rbx
  unsigned __int64 v57; // r13
  unsigned __int64 v58; // r14
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // r15
  __int64 v61; // rbx
  unsigned __int64 v62; // r13
  unsigned __int64 v63; // r14
  unsigned __int64 v64; // rdi
  unsigned __int64 v65; // rdi
  unsigned int v67; // eax
  _QWORD *v68; // rdi
  int v69; // r13d
  _QWORD *v70; // rax
  unsigned int v71; // eax
  _QWORD *v72; // rax
  __int64 v73; // rdx
  _QWORD *i; // rdx
  __int64 *v75; // [rsp+0h] [rbp-40h]
  __int64 *v76; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = &unk_4A20678;
  v2 = *(volatile signed __int32 **)(a1 + 1272);
  if ( v2 && !_InterlockedSub(v2 + 2, 1u) )
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v2 + 8LL))(v2);
  sub_2240A30((unsigned __int64 *)(a1 + 1240));
  sub_2240A30((unsigned __int64 *)(a1 + 1208));
  v3 = *(_QWORD *)(a1 + 1192);
  if ( v3 )
  {
    sub_C7D6A0(*(_QWORD *)(v3 + 8), 24LL * *(unsigned int *)(v3 + 24), 8);
    j_j___libc_free_0(v3);
  }
  sub_26BCC80(*(_QWORD *)(a1 + 1160));
  v4 = *(_QWORD *)(a1 + 1136);
  if ( v4 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  v5 = *(unsigned int *)(a1 + 1112);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD **)(a1 + 1096);
    v7 = &v6[7 * v5];
    do
    {
      if ( *v6 != -4096 && *v6 != -8192 )
      {
        v8 = v6[3];
        while ( v8 )
        {
          sub_26BBA00(*(_QWORD *)(v8 + 24));
          v9 = v8;
          v8 = *(_QWORD *)(v8 + 16);
          j_j___libc_free_0(v9);
        }
      }
      v6 += 7;
    }
    while ( v7 != v6 );
    v5 = *(unsigned int *)(a1 + 1112);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1096), 56 * v5, 8);
  v10 = *(unsigned int *)(a1 + 1080);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 1064);
    v12 = &v11[11 * v10];
    do
    {
      if ( *v11 != -4096 && *v11 != -8192 )
      {
        v13 = v11[1];
        if ( (_QWORD *)v13 != v11 + 3 )
          _libc_free(v13);
      }
      v11 += 11;
    }
    while ( v12 != v11 );
    v10 = *(unsigned int *)(a1 + 1080);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1064), 88 * v10, 8);
  v14 = *(unsigned int *)(a1 + 1048);
  if ( (_DWORD)v14 )
  {
    v15 = *(_QWORD **)(a1 + 1032);
    v16 = &v15[11 * v14];
    do
    {
      if ( *v15 != -8192 && *v15 != -4096 )
      {
        v17 = v15[1];
        if ( (_QWORD *)v17 != v15 + 3 )
          _libc_free(v17);
      }
      v15 += 11;
    }
    while ( v16 != v15 );
    v14 = *(unsigned int *)(a1 + 1048);
  }
  v18 = 88 * v14;
  sub_C7D6A0(*(_QWORD *)(a1 + 1032), 88 * v14, 8);
  v19 = *(_QWORD *)(a1 + 1016);
  if ( v19 )
  {
    v20 = *(_DWORD *)(v19 + 16);
    ++*(_QWORD *)v19;
    if ( v20 )
    {
      v21 = 4 * v20;
      v18 = 64;
      v22 = *(unsigned int *)(v19 + 24);
      if ( (unsigned int)(4 * v20) < 0x40 )
        v21 = 64;
      if ( (unsigned int)v22 <= v21 )
        goto LABEL_37;
      v67 = v20 - 1;
      if ( v67 )
      {
        _BitScanReverse(&v67, v67);
        v68 = *(_QWORD **)(v19 + 8);
        v69 = 1 << (33 - (v67 ^ 0x1F));
        if ( v69 < 64 )
          v69 = 64;
        if ( (_DWORD)v22 == v69 )
        {
          *(_QWORD *)(v19 + 16) = 0;
          v70 = &v68[2 * (unsigned int)v22];
          do
          {
            if ( v68 )
              *v68 = -4096;
            v68 += 2;
          }
          while ( v70 != v68 );
          goto LABEL_40;
        }
      }
      else
      {
        v68 = *(_QWORD **)(v19 + 8);
        v69 = 64;
      }
      v18 = 16 * v22;
      sub_C7D6A0((__int64)v68, 16 * v22, 8);
      v71 = sub_26BC060(v69);
      *(_DWORD *)(v19 + 24) = v71;
      if ( v71 )
      {
        v18 = 8;
        v72 = (_QWORD *)sub_C7D670(16LL * v71, 8);
        v73 = *(unsigned int *)(v19 + 24);
        *(_QWORD *)(v19 + 16) = 0;
        *(_QWORD *)(v19 + 8) = v72;
        for ( i = &v72[2 * v73]; i != v72; v72 += 2 )
        {
          if ( v72 )
            *v72 = -4096;
        }
LABEL_40:
        v75 = *(__int64 **)(v19 + 40);
        if ( *(__int64 **)(v19 + 32) != v75 )
        {
          v76 = *(__int64 **)(v19 + 32);
          do
          {
            v25 = *v76;
            v26 = *(__int64 **)(*v76 + 16);
            if ( *(__int64 **)(*v76 + 8) == v26 )
            {
              *(_BYTE *)(v25 + 152) = 1;
            }
            else
            {
              v27 = *(__int64 **)(*v76 + 8);
              do
              {
                v28 = *v27++;
                sub_D47BB0(v28, v18);
              }
              while ( v26 != v27 );
              *(_BYTE *)(v25 + 152) = 1;
              v29 = *(_QWORD *)(v25 + 8);
              if ( v29 != *(_QWORD *)(v25 + 16) )
                *(_QWORD *)(v25 + 16) = v29;
            }
            v30 = *(_QWORD *)(v25 + 32);
            if ( v30 != *(_QWORD *)(v25 + 40) )
              *(_QWORD *)(v25 + 40) = v30;
            ++*(_QWORD *)(v25 + 56);
            if ( *(_BYTE *)(v25 + 84) )
            {
              *(_QWORD *)v25 = 0;
            }
            else
            {
              v31 = 4 * (*(_DWORD *)(v25 + 76) - *(_DWORD *)(v25 + 80));
              v32 = *(unsigned int *)(v25 + 72);
              if ( v31 < 0x20 )
                v31 = 32;
              if ( (unsigned int)v32 > v31 )
              {
                sub_C8C990(v25 + 56, v18);
              }
              else
              {
                v18 = 0xFFFFFFFFLL;
                memset(*(void **)(v25 + 64), -1, 8 * v32);
              }
              v33 = *(_BYTE *)(v25 + 84);
              *(_QWORD *)v25 = 0;
              if ( !v33 )
                _libc_free(*(_QWORD *)(v25 + 64));
            }
            v34 = *(_QWORD *)(v25 + 32);
            if ( v34 )
            {
              v18 = *(_QWORD *)(v25 + 48) - v34;
              j_j___libc_free_0(v34);
            }
            v35 = *(_QWORD *)(v25 + 8);
            if ( v35 )
            {
              v18 = *(_QWORD *)(v25 + 24) - v35;
              j_j___libc_free_0(v35);
            }
            ++v76;
          }
          while ( v75 != v76 );
          v36 = *(_QWORD *)(v19 + 32);
          if ( *(_QWORD *)(v19 + 40) != v36 )
            *(_QWORD *)(v19 + 40) = v36;
        }
        v37 = *(__int64 **)(v19 + 120);
        v38 = &v37[2 * *(unsigned int *)(v19 + 128)];
        while ( v38 != v37 )
        {
          v39 = v37[1];
          v40 = *v37;
          v37 += 2;
          sub_C7D6A0(v40, v39, 16);
        }
        *(_DWORD *)(v19 + 128) = 0;
        v41 = *(unsigned int *)(v19 + 80);
        if ( (_DWORD)v41 )
        {
          *(_QWORD *)(v19 + 136) = 0;
          v42 = *(__int64 **)(v19 + 72);
          v43 = *v42;
          v44 = &v42[v41];
          v45 = v42 + 1;
          *(_QWORD *)(v19 + 56) = *v42;
          for ( *(_QWORD *)(v19 + 64) = v43 + 4096; v44 != v45; v42 = *(__int64 **)(v19 + 72) )
          {
            v46 = *v45;
            v47 = (unsigned int)(v45 - v42) >> 7;
            v48 = 4096LL << v47;
            if ( v47 >= 0x1E )
              v48 = 0x40000000000LL;
            ++v45;
            sub_C7D6A0(v46, v48, 16);
          }
          *(_DWORD *)(v19 + 80) = 1;
          sub_C7D6A0(*v42, 4096, 16);
          v49 = *(__int64 **)(v19 + 120);
          v50 = (unsigned __int64)&v49[2 * *(unsigned int *)(v19 + 128)];
          if ( v49 == (__int64 *)v50 )
            goto LABEL_73;
          do
          {
            v51 = v49[1];
            v52 = *v49;
            v49 += 2;
            sub_C7D6A0(v52, v51, 16);
          }
          while ( (__int64 *)v50 != v49 );
        }
        v50 = *(_QWORD *)(v19 + 120);
LABEL_73:
        if ( v50 != v19 + 136 )
          _libc_free(v50);
        v53 = *(_QWORD *)(v19 + 72);
        if ( v53 != v19 + 88 )
          _libc_free(v53);
        v54 = *(_QWORD *)(v19 + 32);
        if ( v54 )
          j_j___libc_free_0(v54);
        sub_C7D6A0(*(_QWORD *)(v19 + 8), 16LL * *(unsigned int *)(v19 + 24), 8);
        j_j___libc_free_0(v19);
        goto LABEL_80;
      }
    }
    else
    {
      if ( !*(_DWORD *)(v19 + 20) )
        goto LABEL_40;
      v22 = *(unsigned int *)(v19 + 24);
      if ( (unsigned int)v22 <= 0x40 )
      {
LABEL_37:
        v23 = *(_QWORD **)(v19 + 8);
        for ( j = &v23[2 * v22]; j != v23; v23 += 2 )
          *v23 = -4096;
        goto LABEL_39;
      }
      v18 = 16 * v22;
      sub_C7D6A0(*(_QWORD *)(v19 + 8), 16 * v22, 8);
      *(_DWORD *)(v19 + 24) = 0;
    }
    *(_QWORD *)(v19 + 8) = 0;
LABEL_39:
    *(_QWORD *)(v19 + 16) = 0;
    goto LABEL_40;
  }
LABEL_80:
  v55 = *(_QWORD *)(a1 + 1008);
  if ( v55 )
  {
    v56 = *(_QWORD *)(v55 + 48);
    v57 = v56 + 8LL * *(unsigned int *)(v55 + 56);
    if ( v56 != v57 )
    {
      do
      {
        v58 = *(_QWORD *)(v57 - 8);
        v57 -= 8LL;
        if ( v58 )
        {
          v59 = *(_QWORD *)(v58 + 24);
          if ( v59 != v58 + 40 )
            _libc_free(v59);
          j_j___libc_free_0(v58);
        }
      }
      while ( v56 != v57 );
      v57 = *(_QWORD *)(v55 + 48);
    }
    if ( v57 != v55 + 64 )
      _libc_free(v57);
    if ( *(_QWORD *)v55 != v55 + 16 )
      _libc_free(*(_QWORD *)v55);
    j_j___libc_free_0(v55);
  }
  v60 = *(_QWORD *)(a1 + 1000);
  if ( v60 )
  {
    v61 = *(_QWORD *)(v60 + 24);
    v62 = v61 + 8LL * *(unsigned int *)(v60 + 32);
    if ( v61 != v62 )
    {
      do
      {
        v63 = *(_QWORD *)(v62 - 8);
        v62 -= 8LL;
        if ( v63 )
        {
          v64 = *(_QWORD *)(v63 + 24);
          if ( v64 != v63 + 40 )
            _libc_free(v64);
          j_j___libc_free_0(v63);
        }
      }
      while ( v61 != v62 );
      v62 = *(_QWORD *)(v60 + 24);
    }
    if ( v62 != v60 + 40 )
      _libc_free(v62);
    if ( *(_QWORD *)v60 != v60 + 16 )
      _libc_free(*(_QWORD *)v60);
    j_j___libc_free_0(v60);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 976), 16LL * *(unsigned int *)(a1 + 992), 8);
  sub_26BBBD0(*(_QWORD *)(a1 + 936));
  v65 = *(_QWORD *)(a1 + 392);
  if ( v65 != a1 + 408 )
    _libc_free(v65);
  if ( !*(_BYTE *)(a1 + 132) )
    _libc_free(*(_QWORD *)(a1 + 112));
  sub_C7D6A0(*(_QWORD *)(a1 + 80), 24LL * *(unsigned int *)(a1 + 96), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 48), 16LL * *(unsigned int *)(a1 + 64), 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 32), 8);
}
