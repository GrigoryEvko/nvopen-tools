// Function: sub_2DF7D50
// Address: 0x2df7d50
//
__int64 __fastcall sub_2DF7D50(__int64 a1, unsigned int *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  unsigned int v8; // r10d
  __int64 v9; // r14
  __int64 v10; // r15
  _QWORD *v11; // rax
  __int64 result; // rax
  unsigned int v13; // r12d
  unsigned int i; // r13d
  __int64 v15; // r15
  _QWORD *v16; // rdx
  _QWORD *v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rdx
  unsigned __int8 v20; // di
  __int64 v21; // rsi
  __int64 v22; // rcx
  char v23; // al
  char v24; // al
  _QWORD *v25; // rax
  unsigned __int8 v26; // cl
  __int64 v27; // rax
  __int64 v28; // rsi
  char v29; // al
  char v30; // al
  const void **v31; // rsi
  void *v32; // r11
  unsigned __int64 v33; // rdi
  size_t v34; // rdx
  __int64 v35; // r15
  unsigned int v36; // r12d
  bool v37; // al
  __int64 v38; // rdx
  _QWORD *v39; // r15
  __int64 v40; // r12
  __int64 v41; // r12
  unsigned int v42; // r14d
  __int64 v43; // r15
  _QWORD *v44; // rdx
  _QWORD *v45; // rax
  __int64 v46; // rax
  unsigned __int64 *v47; // r12
  __int64 v48; // rcx
  unsigned __int8 v49; // al
  unsigned __int8 v50; // di
  __int64 v51; // rsi
  char v52; // al
  __int64 v53; // rdi
  char v54; // al
  _QWORD *v55; // rax
  unsigned __int64 *v56; // r14
  __int64 v57; // rax
  unsigned __int64 v58; // r8
  void *v59; // rdi
  size_t v60; // rdx
  __int64 v61; // rax
  unsigned __int64 v62; // r8
  void *v63; // rdi
  size_t v64; // rdx
  __int64 v65; // rax
  unsigned __int64 v66; // rdi
  const void **v67; // rcx
  void *v68; // r9
  size_t v69; // rdx
  __int64 v70; // [rsp+8h] [rbp-68h]
  unsigned __int64 *v71; // [rsp+10h] [rbp-60h]
  unsigned __int64 *v72; // [rsp+18h] [rbp-58h]
  unsigned int v73; // [rsp+20h] [rbp-50h]
  __int64 v74; // [rsp+20h] [rbp-50h]
  unsigned int v75; // [rsp+20h] [rbp-50h]
  __int64 v76; // [rsp+20h] [rbp-50h]
  unsigned int v78; // [rsp+28h] [rbp-48h]
  const void **v80; // [rsp+30h] [rbp-40h]

  v6 = a6;
  v8 = *a2;
  if ( !*a2 )
  {
    if ( a3 )
      goto LABEL_3;
LABEL_38:
    v55 = (_QWORD *)(a1 + 16LL * a3);
    *v55 = a4;
    v55[1] = a5;
    v56 = (unsigned __int64 *)(a1 + 24LL * a3 + 64);
    if ( (unsigned __int64 *)v6 == v56 )
      return a3 + 1;
    if ( (*(_BYTE *)(v6 + 8) & 0x3F) != 0 )
    {
      v61 = sub_2207820(4LL * (*(_BYTE *)(v6 + 8) & 0x3F));
      v62 = *v56;
      v63 = (void *)v61;
      *v56 = v61;
      if ( v62 )
      {
        j_j___libc_free_0_0(v62);
        v63 = (void *)*v56;
      }
      v26 = *(_BYTE *)(v6 + 8) & 0x3F;
      v64 = 4LL * v26;
      if ( v64 )
      {
        memmove(v63, *(const void **)v6, v64);
        v26 = *(_BYTE *)(v6 + 8) & 0x3F;
      }
    }
    else
    {
      *v56 = 0;
      v26 = *(_BYTE *)(v6 + 8) & 0x3F;
    }
    v27 = 3LL * a3;
LABEL_17:
    v28 = a1 + 8 * v27;
    v29 = v26 | *(_BYTE *)(v28 + 72) & 0xC0;
    *(_BYTE *)(v28 + 72) = v29;
    v30 = *(_BYTE *)(v6 + 8) & 0x40 | v29 & 0xBF;
    *(_BYTE *)(v28 + 72) = v30;
    *(_BYTE *)(v28 + 72) = *(_BYTE *)(v6 + 8) & 0x80 | v30 & 0x7F;
    *(_QWORD *)(v28 + 80) = *(_QWORD *)(v6 + 16);
    return a3 + 1;
  }
  v35 = v8 - 1;
  v75 = *a2;
  v36 = v8 - 1;
  v37 = sub_2DF4840(a1 + 24 * v35 + 64, a6);
  v8 = v75;
  if ( v37 )
  {
    v38 = 16 * v35;
    v39 = (_QWORD *)(a1 + 16 * v35 + 8);
    if ( *v39 == a4 )
    {
      *a2 = v36;
      if ( v75 != a3
        && (v40 = v75, v76 = v38, v78 = v8, sub_2DF4840(a1 + 24LL * v8 + 64, v6))
        && (v41 = 16 * v40, *(_QWORD *)(a1 + v41) == a5) )
      {
        v42 = v78 + 1;
        for ( *(_QWORD *)(a1 + v76 + 8) = *(_QWORD *)(a1 + v41 + 8); a3 != v42; ++v42 )
        {
          v43 = v42 - 1;
          v44 = (_QWORD *)(a1 + 16LL * v42);
          v45 = (_QWORD *)(a1 + 16 * v43);
          *v45 = *v44;
          v45[1] = v44[1];
          v46 = 24LL * v42;
          v47 = (unsigned __int64 *)(a1 + 24 * v43 + 64);
          v48 = a1 + v46 + 64;
          if ( (unsigned __int64 *)v48 != v47 )
          {
            v49 = *(_BYTE *)(a1 + v46 + 72) & 0x3F;
            if ( (*(_BYTE *)(v48 + 8) & 0x3F) != 0 )
            {
              v80 = (const void **)v48;
              v65 = sub_2207820(4LL * v49);
              v66 = *v47;
              v67 = v80;
              *v47 = v65;
              v68 = (void *)v65;
              if ( v66 )
              {
                j_j___libc_free_0_0(v66);
                v68 = (void *)*v47;
                v67 = v80;
              }
              v50 = *(_BYTE *)(a1 + 24LL * v42 + 72) & 0x3F;
              v69 = 4LL * v50;
              if ( v69 )
              {
                memmove(v68, *v67, v69);
                v50 = *(_BYTE *)(a1 + 24LL * v42 + 72) & 0x3F;
              }
            }
            else
            {
              *v47 = 0;
              v50 = *(_BYTE *)(v48 + 8) & 0x3F;
            }
            v51 = a1 + 24 * v43;
            v52 = v50 | *(_BYTE *)(v51 + 72) & 0xC0;
            v53 = a1 + 24LL * v42;
            *(_BYTE *)(v51 + 72) = v52;
            v54 = *(_BYTE *)(v53 + 72) & 0x40 | v52 & 0xBF;
            *(_BYTE *)(v51 + 72) = v54;
            *(_BYTE *)(v51 + 72) = *(_BYTE *)(v53 + 72) & 0x80 | v54 & 0x7F;
            *(_QWORD *)(v51 + 80) = *(_QWORD *)(v53 + 80);
          }
        }
        return a3 - 1;
      }
      else
      {
        *v39 = a5;
        return a3;
      }
    }
  }
  result = 5;
  if ( v75 == 4 )
    return result;
  if ( v75 == a3 )
    goto LABEL_38;
LABEL_3:
  v9 = v8;
  v73 = v8;
  v10 = a1 + 24LL * v8 + 64;
  if ( sub_2DF4840(v10, v6) )
  {
    v11 = (_QWORD *)(a1 + 16 * v9);
    if ( *v11 == a5 )
    {
      *v11 = a4;
      return a3;
    }
  }
  result = 5;
  if ( a3 != 4 )
  {
    v71 = (unsigned __int64 *)v10;
    v13 = v73;
    v70 = v6;
    for ( i = a3 - 1; ; --i )
    {
      v15 = i + 1;
      v16 = (_QWORD *)(a1 + 16LL * i);
      v17 = (_QWORD *)(a1 + 16 * v15);
      *v17 = *v16;
      v17[1] = v16[1];
      v18 = 24LL * i;
      v19 = (_QWORD *)(a1 + 24 * v15 + 64);
      if ( (_QWORD *)(a1 + v18 + 64) != v19 )
      {
        if ( (*(_BYTE *)(a1 + v18 + 72) & 0x3F) != 0 )
        {
          v72 = (unsigned __int64 *)(a1 + 24 * v15 + 64);
          v74 = a1 + v18 + 64;
          v31 = (const void **)v74;
          v32 = (void *)sub_2207820(4LL * (*(_BYTE *)(a1 + v18 + 72) & 0x3F));
          v33 = *v72;
          *v72 = (unsigned __int64)v32;
          if ( v33 )
          {
            j_j___libc_free_0_0(v33);
            v31 = (const void **)v74;
            v32 = (void *)*v72;
          }
          v20 = *(_BYTE *)(a1 + 24LL * i + 72) & 0x3F;
          v34 = 4LL * v20;
          if ( v34 )
          {
            memmove(v32, *v31, v34);
            v20 = *(_BYTE *)(a1 + 24LL * i + 72) & 0x3F;
          }
        }
        else
        {
          *v19 = 0;
          v20 = *(_BYTE *)(a1 + v18 + 72) & 0x3F;
        }
        v21 = a1 + 24 * v15;
        v22 = a1 + 24LL * i;
        v23 = v20 | *(_BYTE *)(v21 + 72) & 0xC0;
        *(_BYTE *)(v21 + 72) = v23;
        v24 = *(_BYTE *)(v22 + 72) & 0x40 | v23 & 0xBF;
        *(_BYTE *)(v21 + 72) = v24;
        *(_BYTE *)(v21 + 72) = *(_BYTE *)(v22 + 72) & 0x80 | v24 & 0x7F;
        *(_QWORD *)(v21 + 80) = *(_QWORD *)(v22 + 80);
      }
      if ( v13 == i )
        break;
    }
    v6 = v70;
    v25 = (_QWORD *)(a1 + 16 * v9);
    *v25 = a4;
    v25[1] = a5;
    if ( (unsigned __int64 *)v70 == v71 )
      return a3 + 1;
    if ( (*(_BYTE *)(v70 + 8) & 0x3F) != 0 )
    {
      v57 = sub_2207820(4LL * (*(_BYTE *)(v70 + 8) & 0x3F));
      v58 = *v71;
      v59 = (void *)v57;
      *v71 = v57;
      if ( v58 )
      {
        j_j___libc_free_0_0(v58);
        v59 = (void *)*v71;
      }
      v26 = *(_BYTE *)(v70 + 8) & 0x3F;
      v60 = 4LL * v26;
      if ( v60 )
      {
        memmove(v59, *(const void **)v70, v60);
        v26 = *(_BYTE *)(v70 + 8) & 0x3F;
      }
    }
    else
    {
      *v71 = 0;
      v26 = *(_BYTE *)(v70 + 8) & 0x3F;
    }
    v27 = 3 * v9;
    goto LABEL_17;
  }
  return result;
}
