// Function: sub_9BACB0
// Address: 0x9bacb0
//
__int64 __fastcall sub_9BACB0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  _QWORD *v11; // rax
  _QWORD *i; // rdx
  __int64 v13; // rax
  __int64 v14; // rbx
  char *v15; // rax
  char *v16; // r12
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // r14
  unsigned __int8 v20; // al
  __int64 v21; // r13
  const __m128i *v22; // r12
  char v23; // bl
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r13
  __int64 v32; // r11
  unsigned __int64 v33; // rax
  char v34; // r10
  unsigned int v35; // esi
  __int64 v36; // r8
  __int64 *v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 *v42; // r9
  int v43; // ecx
  __int64 v44; // rax
  unsigned __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // rdx
  __m128i *v48; // rax
  int v49; // r8d
  int v50; // r8d
  __int64 v51; // rdi
  unsigned int v52; // esi
  __int64 v53; // rax
  __int64 *v54; // rdx
  unsigned __int64 v55; // r13
  __int64 v56; // rdi
  __int64 v57; // rsi
  int v58; // r8d
  int v59; // r8d
  __int64 v60; // rdi
  unsigned int v61; // esi
  __int64 v62; // rax
  __int64 *v63; // [rsp+0h] [rbp-F0h]
  unsigned int v64; // [rsp+0h] [rbp-F0h]
  int v65; // [rsp+8h] [rbp-E8h]
  __int64 v66; // [rsp+8h] [rbp-E8h]
  __int64 v67; // [rsp+8h] [rbp-E8h]
  __int64 v68; // [rsp+8h] [rbp-E8h]
  char *v69; // [rsp+10h] [rbp-E0h]
  char *v70; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v73; // [rsp+30h] [rbp-C0h]
  __int64 v74; // [rsp+38h] [rbp-B8h]
  unsigned int v75; // [rsp+38h] [rbp-B8h]
  char v76; // [rsp+38h] [rbp-B8h]
  int v77; // [rsp+38h] [rbp-B8h]
  char v78; // [rsp+38h] [rbp-B8h]
  char v79; // [rsp+38h] [rbp-B8h]
  int v80; // [rsp+38h] [rbp-B8h]
  __int64 v82; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v83; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v84; // [rsp+58h] [rbp-98h]
  __int64 v85; // [rsp+60h] [rbp-90h]
  __int64 v86; // [rsp+68h] [rbp-88h]
  char v87; // [rsp+70h] [rbp-80h]
  _QWORD v88[2]; // [rsp+80h] [rbp-70h] BYREF
  _QWORD *v89; // [rsp+90h] [rbp-60h]
  __int64 v90; // [rsp+98h] [rbp-58h]
  unsigned int v91; // [rsp+A0h] [rbp-50h]
  void *src; // [rsp+A8h] [rbp-48h]
  char *v93; // [rsp+B0h] [rbp-40h]
  char *v94; // [rsp+B8h] [rbp-38h]

  v3 = sub_AA4E30(**(_QWORD **)(a1[1] + 32LL));
  v4 = a1[1];
  v88[1] = 0;
  v5 = v3;
  v6 = *(_QWORD *)(v4 + 40) - *(_QWORD *)(v4 + 32);
  v88[0] = v4;
  v7 = (unsigned int)(v6 >> 3) | ((unsigned __int64)(unsigned int)(v6 >> 3) >> 1);
  v8 = (((v7 >> 2) | v7) >> 4) | (v7 >> 2) | v7;
  v9 = (((v8 >> 8) | v8) >> 16) | (v8 >> 8) | v8;
  if ( (_DWORD)v9 == -1 )
  {
    v89 = 0;
    v90 = 0;
    v91 = 0;
  }
  else
  {
    v10 = 4 * ((int)v9 + 1) / 3u + 1;
    v91 = (((((((((v10 | (v10 >> 1)) >> 2) | v10 | (v10 >> 1)) >> 4) | ((v10 | (v10 >> 1)) >> 2) | v10 | (v10 >> 1)) >> 8)
           | ((((v10 | (v10 >> 1)) >> 2) | v10 | (v10 >> 1)) >> 4)
           | ((v10 | (v10 >> 1)) >> 2)
           | v10
           | (v10 >> 1)) >> 16)
         | ((((((v10 | (v10 >> 1)) >> 2) | v10 | (v10 >> 1)) >> 4) | ((v10 | (v10 >> 1)) >> 2) | v10 | (v10 >> 1)) >> 8)
         | ((((v10 | (v10 >> 1)) >> 2) | v10 | (v10 >> 1)) >> 4)
         | ((v10 | (v10 >> 1)) >> 2)
         | v10
         | (v10 >> 1))
        + 1;
    v11 = (_QWORD *)sub_C7D670(
                      16
                    * ((((((((((v10 | (v10 >> 1)) >> 2) | v10 | (v10 >> 1)) >> 4)
                          | ((v10 | (v10 >> 1)) >> 2)
                          | v10
                          | (v10 >> 1)) >> 8)
                        | ((((v10 | (v10 >> 1)) >> 2) | v10 | (v10 >> 1)) >> 4)
                        | ((v10 | (v10 >> 1)) >> 2)
                        | v10
                        | (v10 >> 1)) >> 16)
                      | ((((((v10 | (v10 >> 1)) >> 2) | v10 | (v10 >> 1)) >> 4)
                        | ((v10 | (v10 >> 1)) >> 2)
                        | v10
                        | (v10 >> 1)) >> 8)
                      | ((((v10 | (v10 >> 1)) >> 2) | v10 | (v10 >> 1)) >> 4)
                      | ((v10 | (v10 >> 1)) >> 2)
                      | v10
                      | (v10 >> 1))
                     + 1),
                      8);
    v90 = 0;
    v89 = v11;
    for ( i = &v11[2 * v91]; i != v11; v11 += 2 )
    {
      if ( v11 )
        *v11 = -4096;
    }
  }
  src = 0;
  v93 = 0;
  v94 = 0;
  v13 = (__int64)(*(_QWORD *)(v4 + 40) - *(_QWORD *)(v4 + 32)) >> 3;
  if ( (_DWORD)v13 )
  {
    v14 = 8LL * (unsigned int)v13;
    v15 = (char *)sub_22077B0(v14);
    v16 = v15;
    if ( v93 - (_BYTE *)src > 0 )
    {
      memmove(v15, src, v93 - (_BYTE *)src);
      j_j___libc_free_0(src, v94 - (_BYTE *)src);
    }
    src = v16;
    v93 = v16;
    v94 = &v16[v14];
  }
  sub_D4E470(v88, a1[3]);
  v69 = (char *)src;
  v70 = v93;
  if ( v93 != src )
  {
    v17 = v5;
    while ( 1 )
    {
      v18 = *((_QWORD *)v70 - 1);
      v82 = v18 + 48;
      v19 = *(_QWORD *)(v18 + 56);
      if ( v18 + 48 != v19 )
        break;
LABEL_28:
      v70 -= 8;
      if ( v69 == v70 )
      {
        v69 = (char *)src;
        goto LABEL_30;
      }
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v19 )
          BUG();
        v20 = *(_BYTE *)(v19 - 24);
        if ( v20 <= 0x1Cu )
          goto LABEL_15;
        if ( v20 == 61 )
          break;
        if ( v20 != 62 )
          goto LABEL_15;
        v74 = *(_QWORD *)(v19 - 56);
        if ( !v74 )
          goto LABEL_15;
        v21 = *(_QWORD *)(*(_QWORD *)(v19 - 88) + 8LL);
LABEL_21:
        v22 = (const __m128i *)&v83;
        v23 = sub_AE5020(v17, v21);
        v24 = sub_9208B0(v17, v21);
        v84 = v25;
        v83 = ((1LL << v23) + ((unsigned __int64)(v24 + 7) >> 3) - 1) >> v23 << v23;
        v26 = sub_CA1930(&v83);
        v27 = sub_9208B0(v17, v21);
        v84 = v28;
        v83 = v27;
        if ( 8 * v26 != sub_CA1930(&v83) )
          goto LABEL_15;
        v29 = sub_D34EB0(*a1, v21, v74, a1[1], a3, 1, 0);
        v73 = 0;
        v84 = v30;
        v83 = v29;
        if ( (_BYTE)v30 )
          v73 = v83;
        v31 = v19 - 24;
        v32 = sub_D34370(*a1, a3, v74);
        _BitScanReverse64(&v33, 1LL << (*(_WORD *)(v19 - 22) >> 1));
        v34 = 63 - (v33 ^ 0x3F);
        v35 = *(_DWORD *)(a2 + 24);
        if ( !v35 )
        {
          ++*(_QWORD *)a2;
          goto LABEL_48;
        }
        v36 = *(_QWORD *)(a2 + 8);
        v75 = (v35 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
        v37 = (__int64 *)(v36 + 16LL * v75);
        v38 = *v37;
        if ( v31 != *v37 )
        {
          v65 = 1;
          v42 = 0;
          while ( v38 != -4096 )
          {
            if ( v38 == -8192 && !v42 )
              v42 = v37;
            v75 = (v35 - 1) & (v65 + v75);
            v37 = (__int64 *)(v36 + 16LL * v75);
            v38 = *v37;
            if ( v31 == *v37 )
              goto LABEL_26;
            ++v65;
          }
          if ( !v42 )
            v42 = v37;
          ++*(_QWORD *)a2;
          v43 = *(_DWORD *)(a2 + 16) + 1;
          if ( 4 * v43 >= 3 * v35 )
          {
LABEL_48:
            v66 = v32;
            v76 = v34;
            sub_9BAAD0(a2, 2 * v35);
            v49 = *(_DWORD *)(a2 + 24);
            if ( !v49 )
              goto LABEL_79;
            v50 = v49 - 1;
            v51 = *(_QWORD *)(a2 + 8);
            v32 = v66;
            v34 = v76;
            v52 = v50 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
            v43 = *(_DWORD *)(a2 + 16) + 1;
            v42 = (__int64 *)(v51 + 16LL * v52);
            v53 = *v42;
            if ( v31 != *v42 )
            {
              v77 = 1;
              v54 = 0;
              while ( v53 != -4096 )
              {
                if ( !v54 && v53 == -8192 )
                  v54 = v42;
                v52 = v50 & (v77 + v52);
                v42 = (__int64 *)(v51 + 16LL * v52);
                v53 = *v42;
                if ( v31 == *v42 )
                  goto LABEL_41;
                ++v77;
              }
LABEL_52:
              if ( v54 )
                v42 = v54;
            }
          }
          else if ( v35 - *(_DWORD *)(a2 + 20) - v43 <= v35 >> 3 )
          {
            v64 = ((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4);
            v68 = v32;
            v79 = v34;
            sub_9BAAD0(a2, v35);
            v58 = *(_DWORD *)(a2 + 24);
            if ( !v58 )
            {
LABEL_79:
              ++*(_DWORD *)(a2 + 16);
              BUG();
            }
            v59 = v58 - 1;
            v60 = *(_QWORD *)(a2 + 8);
            v34 = v79;
            v32 = v68;
            v61 = v59 & v64;
            v43 = *(_DWORD *)(a2 + 16) + 1;
            v42 = (__int64 *)(v60 + 16LL * (v59 & v64));
            v62 = *v42;
            if ( v31 != *v42 )
            {
              v80 = 1;
              v54 = 0;
              while ( v62 != -4096 )
              {
                if ( !v54 && v62 == -8192 )
                  v54 = v42;
                v61 = v59 & (v80 + v61);
                v42 = (__int64 *)(v60 + 16LL * v61);
                v62 = *v42;
                if ( v31 == *v42 )
                  goto LABEL_41;
                ++v80;
              }
              goto LABEL_52;
            }
          }
LABEL_41:
          *(_DWORD *)(a2 + 16) = v43;
          if ( *v42 != -4096 )
            --*(_DWORD *)(a2 + 20);
          *v42 = v31;
          *((_DWORD *)v42 + 2) = 0;
          v44 = *(unsigned int *)(a2 + 40);
          v45 = *(unsigned int *)(a2 + 44);
          v83 = v19 - 24;
          v84 = 0;
          v46 = v44 + 1;
          v87 = 0;
          v85 = 0;
          v86 = 0;
          if ( v44 + 1 > v45 )
          {
            v63 = v42;
            v67 = v32;
            v55 = *(_QWORD *)(a2 + 32);
            v56 = a2 + 32;
            v78 = v34;
            v57 = a2 + 48;
            if ( v55 > (unsigned __int64)&v83 || (unsigned __int64)&v83 >= v55 + 40 * v44 )
            {
              sub_C8D5F0(v56, v57, v46, 40);
              v42 = v63;
              v32 = v67;
              v34 = v78;
              v47 = *(_QWORD *)(a2 + 32);
              v44 = *(unsigned int *)(a2 + 40);
            }
            else
            {
              sub_C8D5F0(v56, v57, v46, 40);
              v34 = v78;
              v32 = v67;
              v42 = v63;
              v47 = *(_QWORD *)(a2 + 32);
              v44 = *(unsigned int *)(a2 + 40);
              v22 = (const __m128i *)((char *)&v83 + v47 - v55);
            }
          }
          else
          {
            v47 = *(_QWORD *)(a2 + 32);
          }
          v48 = (__m128i *)(v47 + 40 * v44);
          *v48 = _mm_loadu_si128(v22);
          v48[1] = _mm_loadu_si128(v22 + 1);
          v48[2].m128i_i64[0] = v22[2].m128i_i64[0];
          v39 = *(unsigned int *)(a2 + 40);
          *(_DWORD *)(a2 + 40) = v39 + 1;
          *((_DWORD *)v42 + 2) = v39;
          goto LABEL_27;
        }
LABEL_26:
        v39 = *((unsigned int *)v37 + 2);
LABEL_27:
        v40 = *(_QWORD *)(a2 + 32) + 40 * v39;
        *(_QWORD *)(v40 + 8) = v73;
        *(_QWORD *)(v40 + 16) = v32;
        *(_QWORD *)(v40 + 24) = v26;
        *(_BYTE *)(v40 + 32) = v34;
        v19 = *(_QWORD *)(v19 + 8);
        if ( v82 == v19 )
          goto LABEL_28;
      }
      v74 = *(_QWORD *)(v19 - 56);
      if ( v74 )
      {
        v21 = *(_QWORD *)(v19 - 16);
        goto LABEL_21;
      }
LABEL_15:
      v19 = *(_QWORD *)(v19 + 8);
      if ( v82 == v19 )
        goto LABEL_28;
    }
  }
LABEL_30:
  if ( v69 )
    j_j___libc_free_0(v69, v94 - v69);
  return sub_C7D6A0(v89, 16LL * v91, 8);
}
