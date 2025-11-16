// Function: sub_26C14C0
// Address: 0x26c14c0
//
__int64 __fastcall sub_26C14C0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdi
  unsigned __int64 v5; // r13
  __int64 v6; // r15
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rdi
  _QWORD *v10; // rax
  _QWORD *v11; // r12
  __int64 v12; // rdi
  int v13; // eax
  unsigned __int64 v14; // r15
  __int64 v15; // rbx
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rax
  _QWORD *v20; // rdi
  _QWORD *v21; // rax
  unsigned __int64 v22; // r13
  int v23; // eax
  unsigned int v24; // ecx
  __int64 v25; // rdx
  _QWORD *v26; // rax
  _QWORD *j; // rdx
  __int64 *v28; // r12
  __int64 v29; // r15
  __int64 *v30; // rbx
  __int64 *v31; // r14
  __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned int v35; // eax
  __int64 v36; // rdx
  char v37; // al
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  __int64 v40; // rax
  __int64 *v41; // rbx
  __int64 *v42; // r12
  __int64 v43; // rsi
  __int64 v44; // rdi
  __int64 v45; // rdx
  __int64 *v46; // rax
  __int64 v47; // rcx
  __int64 *v48; // rbx
  __int64 *v49; // r15
  __int64 v50; // rdi
  unsigned int v51; // ecx
  __int64 v52; // rsi
  __int64 *v53; // rbx
  unsigned __int64 v54; // r12
  __int64 v55; // rsi
  __int64 v56; // rdi
  unsigned __int64 v57; // rdi
  unsigned __int64 v58; // rdi
  unsigned int v60; // eax
  _QWORD *v61; // rdi
  int v62; // r12d
  unsigned int v63; // eax
  _QWORD *v64; // rax
  __int64 v65; // rdx
  _QWORD *i; // rdx
  _QWORD *v67; // rax
  __int64 *v68; // [rsp+0h] [rbp-40h]

  v2 = a2;
  v3 = sub_22077B0(0x80u);
  v4 = v3;
  if ( v3 )
  {
    *(_QWORD *)(v3 + 96) = 0;
    *(_QWORD *)v3 = v3 + 16;
    *(_QWORD *)(v3 + 8) = 0x100000000LL;
    *(_QWORD *)(v3 + 24) = v3 + 40;
    *(_QWORD *)(v3 + 32) = 0x600000000LL;
    *(_QWORD *)(v3 + 104) = 0;
    *(_BYTE *)(v3 + 112) = 0;
    *(_QWORD *)(v3 + 116) = 0;
  }
  v5 = a1[125];
  a1[125] = v3;
  if ( v5 )
  {
    v6 = *(_QWORD *)(v5 + 24);
    v7 = v6 + 8LL * *(unsigned int *)(v5 + 32);
    if ( v6 != v7 )
    {
      do
      {
        v8 = *(_QWORD *)(v7 - 8);
        v7 -= 8LL;
        if ( v8 )
        {
          v9 = *(_QWORD *)(v8 + 24);
          if ( v9 != v8 + 40 )
            _libc_free(v9);
          j_j___libc_free_0(v8);
        }
      }
      while ( v6 != v7 );
      v7 = *(_QWORD *)(v5 + 24);
    }
    if ( v7 != v5 + 40 )
      _libc_free(v7);
    if ( *(_QWORD *)v5 != v5 + 16 )
      _libc_free(*(_QWORD *)v5);
    a2 = 128;
    j_j___libc_free_0(v5);
    v4 = a1[125];
  }
  *(_QWORD *)(v4 + 104) = v2;
  *(_DWORD *)(v4 + 120) = *(_DWORD *)(v2 + 92);
  sub_B1F440(v4);
  v10 = (_QWORD *)sub_22077B0(0x98u);
  v11 = v10;
  if ( v10 )
  {
    v12 = (__int64)v10;
    v10[15] = 0;
    *v10 = v10 + 2;
    v10[1] = 0x400000000LL;
    v10[6] = v10 + 8;
    v10[7] = 0x600000000LL;
    v13 = *(_DWORD *)(v2 + 92);
    *((_BYTE *)v11 + 136) = 0;
    *((_DWORD *)v11 + 35) = 0;
    v11[16] = v2;
    *((_DWORD *)v11 + 36) = v13;
    sub_B29120(v12);
  }
  v14 = a1[126];
  a1[126] = (__int64)v11;
  if ( v14 )
  {
    v15 = *(_QWORD *)(v14 + 48);
    v16 = v15 + 8LL * *(unsigned int *)(v14 + 56);
    if ( v15 != v16 )
    {
      do
      {
        v17 = *(_QWORD *)(v16 - 8);
        v16 -= 8LL;
        if ( v17 )
        {
          v18 = *(_QWORD *)(v17 + 24);
          if ( v18 != v17 + 40 )
            _libc_free(v18);
          j_j___libc_free_0(v17);
        }
      }
      while ( v15 != v16 );
      v16 = *(_QWORD *)(v14 + 48);
    }
    if ( v16 != v14 + 64 )
      _libc_free(v16);
    if ( *(_QWORD *)v14 != v14 + 16 )
      _libc_free(*(_QWORD *)v14);
    a2 = 152;
    j_j___libc_free_0(v14);
  }
  v19 = (_QWORD *)sub_22077B0(0x98u);
  v20 = v19;
  if ( v19 )
  {
    *v19 = 0;
    v21 = v19 + 11;
    *(v21 - 10) = 0;
    *(v21 - 9) = 0;
    *((_DWORD *)v21 - 16) = 0;
    *(v21 - 7) = 0;
    *(v21 - 6) = 0;
    *(v21 - 5) = 0;
    *(v21 - 4) = 0;
    *(v21 - 3) = 0;
    v20[9] = v21;
    v20[10] = 0x400000000LL;
    v20[15] = v20 + 17;
    v20[16] = 0;
    v20[17] = 0;
    v20[18] = 1;
  }
  v22 = a1[127];
  a1[127] = (__int64)v20;
  if ( v22 )
  {
    v23 = *(_DWORD *)(v22 + 16);
    ++*(_QWORD *)v22;
    if ( v23 )
    {
      v24 = 4 * v23;
      a2 = 64;
      v25 = *(unsigned int *)(v22 + 24);
      if ( (unsigned int)(4 * v23) < 0x40 )
        v24 = 64;
      if ( (unsigned int)v25 <= v24 )
        goto LABEL_38;
      v60 = v23 - 1;
      if ( v60 )
      {
        _BitScanReverse(&v60, v60);
        v61 = *(_QWORD **)(v22 + 8);
        v62 = 1 << (33 - (v60 ^ 0x1F));
        if ( v62 < 64 )
          v62 = 64;
        if ( v62 == (_DWORD)v25 )
        {
          *(_QWORD *)(v22 + 16) = 0;
          v67 = &v61[2 * (unsigned int)v62];
          do
          {
            if ( v61 )
              *v61 = -4096;
            v61 += 2;
          }
          while ( v67 != v61 );
LABEL_41:
          v28 = *(__int64 **)(v22 + 32);
          v68 = *(__int64 **)(v22 + 40);
          if ( v28 != v68 )
          {
            do
            {
              v29 = *v28;
              v30 = *(__int64 **)(*v28 + 16);
              if ( *(__int64 **)(*v28 + 8) == v30 )
              {
                *(_BYTE *)(v29 + 152) = 1;
              }
              else
              {
                v31 = *(__int64 **)(*v28 + 8);
                do
                {
                  v32 = *v31++;
                  sub_D47BB0(v32, a2);
                }
                while ( v30 != v31 );
                *(_BYTE *)(v29 + 152) = 1;
                v33 = *(_QWORD *)(v29 + 8);
                if ( *(_QWORD *)(v29 + 16) != v33 )
                  *(_QWORD *)(v29 + 16) = v33;
              }
              v34 = *(_QWORD *)(v29 + 32);
              if ( v34 != *(_QWORD *)(v29 + 40) )
                *(_QWORD *)(v29 + 40) = v34;
              ++*(_QWORD *)(v29 + 56);
              if ( *(_BYTE *)(v29 + 84) )
              {
                *(_QWORD *)v29 = 0;
              }
              else
              {
                v35 = 4 * (*(_DWORD *)(v29 + 76) - *(_DWORD *)(v29 + 80));
                v36 = *(unsigned int *)(v29 + 72);
                if ( v35 < 0x20 )
                  v35 = 32;
                if ( (unsigned int)v36 > v35 )
                {
                  sub_C8C990(v29 + 56, a2);
                }
                else
                {
                  a2 = 0xFFFFFFFFLL;
                  memset(*(void **)(v29 + 64), -1, 8 * v36);
                }
                v37 = *(_BYTE *)(v29 + 84);
                *(_QWORD *)v29 = 0;
                if ( !v37 )
                  _libc_free(*(_QWORD *)(v29 + 64));
              }
              v38 = *(_QWORD *)(v29 + 32);
              if ( v38 )
              {
                a2 = *(_QWORD *)(v29 + 48) - v38;
                j_j___libc_free_0(v38);
              }
              v39 = *(_QWORD *)(v29 + 8);
              if ( v39 )
              {
                a2 = *(_QWORD *)(v29 + 24) - v39;
                j_j___libc_free_0(v39);
              }
              ++v28;
            }
            while ( v68 != v28 );
            v40 = *(_QWORD *)(v22 + 32);
            if ( v40 != *(_QWORD *)(v22 + 40) )
              *(_QWORD *)(v22 + 40) = v40;
          }
          v41 = *(__int64 **)(v22 + 120);
          v42 = &v41[2 * *(unsigned int *)(v22 + 128)];
          while ( v42 != v41 )
          {
            v43 = v41[1];
            v44 = *v41;
            v41 += 2;
            sub_C7D6A0(v44, v43, 16);
          }
          *(_DWORD *)(v22 + 128) = 0;
          v45 = *(unsigned int *)(v22 + 80);
          if ( (_DWORD)v45 )
          {
            *(_QWORD *)(v22 + 136) = 0;
            v46 = *(__int64 **)(v22 + 72);
            v47 = *v46;
            v48 = &v46[v45];
            v49 = v46 + 1;
            *(_QWORD *)(v22 + 56) = *v46;
            for ( *(_QWORD *)(v22 + 64) = v47 + 4096; v48 != v49; v46 = *(__int64 **)(v22 + 72) )
            {
              v50 = *v49;
              v51 = (unsigned int)(v49 - v46) >> 7;
              v52 = 4096LL << v51;
              if ( v51 >= 0x1E )
                v52 = 0x40000000000LL;
              ++v49;
              sub_C7D6A0(v50, v52, 16);
            }
            *(_DWORD *)(v22 + 80) = 1;
            sub_C7D6A0(*v46, 4096, 16);
            v53 = *(__int64 **)(v22 + 120);
            v54 = (unsigned __int64)&v53[2 * *(unsigned int *)(v22 + 128)];
            if ( v53 == (__int64 *)v54 )
              goto LABEL_73;
            do
            {
              v55 = v53[1];
              v56 = *v53;
              v53 += 2;
              sub_C7D6A0(v56, v55, 16);
            }
            while ( (__int64 *)v54 != v53 );
          }
          v54 = *(_QWORD *)(v22 + 120);
LABEL_73:
          if ( v54 != v22 + 136 )
            _libc_free(v54);
          v57 = *(_QWORD *)(v22 + 72);
          if ( v57 != v22 + 88 )
            _libc_free(v57);
          v58 = *(_QWORD *)(v22 + 32);
          if ( v58 )
            j_j___libc_free_0(v58);
          sub_C7D6A0(*(_QWORD *)(v22 + 8), 16LL * *(unsigned int *)(v22 + 24), 8);
          j_j___libc_free_0(v22);
          v20 = (_QWORD *)a1[127];
          return sub_D50CB0((__int64)v20, a1[125]);
        }
      }
      else
      {
        v61 = *(_QWORD **)(v22 + 8);
        v62 = 64;
      }
      a2 = 16 * v25;
      sub_C7D6A0((__int64)v61, 16 * v25, 8);
      v63 = sub_26BC060(v62);
      *(_DWORD *)(v22 + 24) = v63;
      if ( v63 )
      {
        a2 = 8;
        v64 = (_QWORD *)sub_C7D670(16LL * v63, 8);
        v65 = *(unsigned int *)(v22 + 24);
        *(_QWORD *)(v22 + 16) = 0;
        *(_QWORD *)(v22 + 8) = v64;
        for ( i = &v64[2 * v65]; i != v64; v64 += 2 )
        {
          if ( v64 )
            *v64 = -4096;
        }
        goto LABEL_41;
      }
    }
    else
    {
      if ( !*(_DWORD *)(v22 + 20) )
        goto LABEL_41;
      v25 = *(unsigned int *)(v22 + 24);
      if ( (unsigned int)v25 <= 0x40 )
      {
LABEL_38:
        v26 = *(_QWORD **)(v22 + 8);
        for ( j = &v26[2 * v25]; j != v26; v26 += 2 )
          *v26 = -4096;
        goto LABEL_40;
      }
      a2 = 16 * v25;
      sub_C7D6A0(*(_QWORD *)(v22 + 8), 16 * v25, 8);
      *(_DWORD *)(v22 + 24) = 0;
    }
    *(_QWORD *)(v22 + 8) = 0;
LABEL_40:
    *(_QWORD *)(v22 + 16) = 0;
    goto LABEL_41;
  }
  return sub_D50CB0((__int64)v20, a1[125]);
}
