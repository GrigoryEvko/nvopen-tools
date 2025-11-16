// Function: sub_15C7550
// Address: 0x15c7550
//
_QWORD *__fastcall sub_15C7550(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, char a6)
{
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned int v11; // esi
  __int64 v12; // rcx
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r10
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // r14
  _BYTE *v20; // r15
  __int64 v21; // r8
  __int64 *v22; // r12
  __int64 v23; // r9
  unsigned int v24; // edi
  _QWORD *v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rbx
  __int64 v28; // rax
  unsigned int v29; // esi
  int v30; // esi
  int v31; // esi
  __int64 v32; // r9
  unsigned int v33; // edx
  int v34; // ecx
  __int64 v35; // rdi
  _QWORD *v37; // r11
  int v38; // ecx
  int v39; // esi
  int v40; // esi
  __int64 v41; // r9
  _QWORD *v42; // r10
  int v43; // r11d
  unsigned int v44; // edx
  __int64 v45; // rdi
  int v46; // r11d
  __int64 *v47; // r8
  int v48; // eax
  int v49; // edx
  int v50; // eax
  int v51; // ecx
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rdi
  int v55; // r10d
  __int64 *v56; // r9
  int v57; // eax
  int v58; // eax
  __int64 v59; // rsi
  int v60; // r9d
  __int64 v61; // r14
  __int64 *v62; // rdi
  __int64 v63; // rcx
  int v64; // r11d
  __int64 v65; // [rsp+0h] [rbp-80h]
  __int64 v67; // [rsp+8h] [rbp-78h]
  int v68; // [rsp+8h] [rbp-78h]
  unsigned int v69; // [rsp+8h] [rbp-78h]
  _BYTE *v71; // [rsp+20h] [rbp-60h] BYREF
  __int64 v72; // [rsp+28h] [rbp-58h]
  _BYTE v73[80]; // [rsp+30h] [rbp-50h] BYREF

  v71 = v73;
  v72 = 0x300000000LL;
  v9 = sub_15C70A0(a2);
  if ( *(_DWORD *)(v9 + 8) == 2 )
  {
    v10 = v9;
    while ( 1 )
    {
      v10 = *(_QWORD *)(v10 - 8);
      if ( !v10 )
        goto LABEL_28;
      v11 = *(_DWORD *)(a5 + 24);
      if ( !v11 )
        break;
      v12 = *(_QWORD *)(a5 + 8);
      v13 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v14 = (__int64 *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( v10 != *v14 )
      {
        v46 = 1;
        v47 = 0;
        while ( v15 != -8 )
        {
          if ( !v47 && v15 == -16 )
            v47 = v14;
          v13 = (v11 - 1) & (v46 + v13);
          v14 = (__int64 *)(v12 + 16LL * v13);
          v15 = *v14;
          if ( v10 == *v14 )
            goto LABEL_6;
          ++v46;
        }
        if ( !v47 )
          v47 = v14;
        v48 = *(_DWORD *)(a5 + 16);
        ++*(_QWORD *)a5;
        v49 = v48 + 1;
        if ( 4 * (v48 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(a5 + 20) - v49 <= v11 >> 3 )
          {
            sub_15C7390(a5, v11);
            v57 = *(_DWORD *)(a5 + 24);
            if ( !v57 )
            {
LABEL_97:
              ++*(_DWORD *)(a5 + 16);
              BUG();
            }
            v58 = v57 - 1;
            v59 = *(_QWORD *)(a5 + 8);
            v60 = 1;
            LODWORD(v61) = v58 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
            v49 = *(_DWORD *)(a5 + 16) + 1;
            v62 = 0;
            v47 = (__int64 *)(v59 + 16LL * (unsigned int)v61);
            v63 = *v47;
            if ( v10 != *v47 )
            {
              while ( v63 != -8 )
              {
                if ( !v62 && v63 == -16 )
                  v62 = v47;
                v61 = v58 & (unsigned int)(v61 + v60);
                v47 = (__int64 *)(v59 + 16 * v61);
                v63 = *v47;
                if ( v10 == *v47 )
                  goto LABEL_52;
                ++v60;
              }
              if ( v62 )
                v47 = v62;
            }
          }
          goto LABEL_52;
        }
LABEL_56:
        sub_15C7390(a5, 2 * v11);
        v50 = *(_DWORD *)(a5 + 24);
        if ( !v50 )
          goto LABEL_97;
        v51 = v50 - 1;
        v52 = *(_QWORD *)(a5 + 8);
        LODWORD(v53) = (v50 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v49 = *(_DWORD *)(a5 + 16) + 1;
        v47 = (__int64 *)(v52 + 16LL * (unsigned int)v53);
        v54 = *v47;
        if ( v10 != *v47 )
        {
          v55 = 1;
          v56 = 0;
          while ( v54 != -8 )
          {
            if ( v54 == -16 && !v56 )
              v56 = v47;
            v53 = v51 & (unsigned int)(v53 + v55);
            v47 = (__int64 *)(v52 + 16 * v53);
            v54 = *v47;
            if ( v10 == *v47 )
              goto LABEL_52;
            ++v55;
          }
          if ( v56 )
            v47 = v56;
        }
LABEL_52:
        *(_DWORD *)(a5 + 16) = v49;
        if ( *v47 != -8 )
          --*(_DWORD *)(a5 + 20);
        *v47 = v10;
        v47[1] = 0;
        goto LABEL_7;
      }
LABEL_6:
      v16 = v14[1];
      if ( v16 )
      {
        v18 = (unsigned int)v72;
        a3 = v16;
        goto LABEL_13;
      }
LABEL_7:
      if ( a6 && (*(_DWORD *)(v10 + 8) != 2 || !*(_QWORD *)(v10 - 8)) )
        goto LABEL_28;
      v17 = (unsigned int)v72;
      if ( (unsigned int)v72 >= HIDWORD(v72) )
      {
        sub_16CD150(&v71, v73, 0, 8);
        v17 = (unsigned int)v72;
      }
      *(_QWORD *)&v71[8 * v17] = v10;
      v18 = (unsigned int)(v72 + 1);
      LODWORD(v72) = v72 + 1;
      if ( *(_DWORD *)(v10 + 8) != 2 )
        goto LABEL_13;
    }
    ++*(_QWORD *)a5;
    goto LABEL_56;
  }
LABEL_28:
  v18 = (unsigned int)v72;
LABEL_13:
  v19 = (unsigned __int64)v71;
  v20 = &v71[8 * v18];
  if ( v71 == v20 )
    goto LABEL_25;
  v21 = a3;
  v22 = a4;
  do
  {
    while ( 1 )
    {
      v27 = *((_QWORD *)v20 - 1);
      v28 = sub_15B9E00(
              v22,
              *(_DWORD *)(v27 + 4),
              *(unsigned __int16 *)(v27 + 2),
              *(_QWORD *)(v27 - 8LL * *(unsigned int *)(v27 + 8)),
              v21,
              1u,
              1);
      v29 = *(_DWORD *)(a5 + 24);
      v21 = v28;
      if ( !v29 )
      {
        ++*(_QWORD *)a5;
LABEL_19:
        v67 = v21;
        sub_15C7390(a5, 2 * v29);
        v30 = *(_DWORD *)(a5 + 24);
        if ( !v30 )
          goto LABEL_98;
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a5 + 8);
        v21 = v67;
        v33 = v31 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v34 = *(_DWORD *)(a5 + 16) + 1;
        v25 = (_QWORD *)(v32 + 16LL * v33);
        v35 = *v25;
        if ( v27 != *v25 )
        {
          v64 = 1;
          v42 = 0;
          while ( v35 != -8 )
          {
            if ( !v42 && v35 == -16 )
              v42 = v25;
            v33 = v31 & (v64 + v33);
            v25 = (_QWORD *)(v32 + 16LL * v33);
            v35 = *v25;
            if ( v27 == *v25 )
              goto LABEL_21;
            ++v64;
          }
LABEL_38:
          if ( v42 )
            v25 = v42;
          goto LABEL_21;
        }
        goto LABEL_21;
      }
      v23 = *(_QWORD *)(a5 + 8);
      v24 = (v29 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v25 = (_QWORD *)(v23 + 16LL * v24);
      v26 = *v25;
      if ( v27 != *v25 )
        break;
LABEL_16:
      v20 -= 8;
      v25[1] = v21;
      if ( (_BYTE *)v19 == v20 )
        goto LABEL_24;
    }
    v68 = 1;
    v37 = 0;
    while ( v26 != -8 )
    {
      if ( !v37 && v26 == -16 )
        v37 = v25;
      v24 = (v29 - 1) & (v68 + v24);
      v25 = (_QWORD *)(v23 + 16LL * v24);
      v26 = *v25;
      if ( v27 == *v25 )
        goto LABEL_16;
      ++v68;
    }
    v38 = *(_DWORD *)(a5 + 16);
    if ( v37 )
      v25 = v37;
    ++*(_QWORD *)a5;
    v34 = v38 + 1;
    if ( 4 * v34 >= 3 * v29 )
      goto LABEL_19;
    if ( v29 - *(_DWORD *)(a5 + 20) - v34 <= v29 >> 3 )
    {
      v65 = v21;
      v69 = ((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4);
      sub_15C7390(a5, v29);
      v39 = *(_DWORD *)(a5 + 24);
      if ( !v39 )
      {
LABEL_98:
        ++*(_DWORD *)(a5 + 16);
        BUG();
      }
      v40 = v39 - 1;
      v41 = *(_QWORD *)(a5 + 8);
      v42 = 0;
      v21 = v65;
      v43 = 1;
      v44 = v40 & v69;
      v34 = *(_DWORD *)(a5 + 16) + 1;
      v25 = (_QWORD *)(v41 + 16LL * (v40 & v69));
      v45 = *v25;
      if ( v27 != *v25 )
      {
        while ( v45 != -8 )
        {
          if ( v45 == -16 && !v42 )
            v42 = v25;
          v44 = v40 & (v43 + v44);
          v25 = (_QWORD *)(v41 + 16LL * v44);
          v45 = *v25;
          if ( v27 == *v25 )
            goto LABEL_21;
          ++v43;
        }
        goto LABEL_38;
      }
    }
LABEL_21:
    *(_DWORD *)(a5 + 16) = v34;
    if ( *v25 != -8 )
      --*(_DWORD *)(a5 + 20);
    v20 -= 8;
    v25[1] = 0;
    *v25 = v27;
    v25[1] = v21;
  }
  while ( (_BYTE *)v19 != v20 );
LABEL_24:
  a3 = v21;
LABEL_25:
  sub_15C7080(a1, a3);
  if ( v71 != v73 )
    _libc_free((unsigned __int64)v71);
  return a1;
}
