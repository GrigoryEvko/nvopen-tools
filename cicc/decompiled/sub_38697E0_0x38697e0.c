// Function: sub_38697E0
// Address: 0x38697e0
//
__int64 *__fastcall sub_38697E0(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // rdx
  __int64 *result; // rax
  __int64 *v7; // rdi
  __int64 v8; // rdx
  _BYTE *v9; // rsi
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  char v16; // di
  __int64 v17; // rcx
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // r12
  __int64 v20; // rax
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  char v24; // si
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // r15
  __int64 v28; // r12
  const char *v29; // rax
  size_t v30; // rdx
  _WORD *v31; // rdi
  char *v32; // rsi
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  unsigned __int64 v35; // r12
  __int64 v36; // r14
  __int64 *v37; // rax
  char v38; // dl
  __int64 v39; // r15
  __int64 *v40; // rax
  __int64 *v41; // rcx
  __int64 *v42; // rsi
  unsigned __int64 v43; // rdx
  char v44; // r8
  char v45; // si
  __int64 v46; // rax
  __int64 v47; // rax
  size_t v48; // [rsp+0h] [rbp-2A0h]
  __int64 *v49; // [rsp+20h] [rbp-280h]
  __int64 *v50; // [rsp+28h] [rbp-278h]
  __int64 v52; // [rsp+48h] [rbp-258h] BYREF
  __int64 v53; // [rsp+50h] [rbp-250h] BYREF
  char v54; // [rsp+60h] [rbp-240h]
  __int64 v55; // [rsp+70h] [rbp-230h] BYREF
  __int64 *v56; // [rsp+78h] [rbp-228h]
  __int64 *v57; // [rsp+80h] [rbp-220h]
  unsigned int v58; // [rsp+88h] [rbp-218h]
  unsigned int v59; // [rsp+8Ch] [rbp-214h]
  int v60; // [rsp+90h] [rbp-210h]
  char v61[64]; // [rsp+98h] [rbp-208h] BYREF
  unsigned __int64 v62; // [rsp+D8h] [rbp-1C8h] BYREF
  unsigned __int64 v63; // [rsp+E0h] [rbp-1C0h]
  unsigned __int64 v64; // [rsp+E8h] [rbp-1B8h]
  _QWORD v65[2]; // [rsp+F0h] [rbp-1B0h] BYREF
  unsigned __int64 v66; // [rsp+100h] [rbp-1A0h]
  _BYTE v67[64]; // [rsp+118h] [rbp-188h] BYREF
  unsigned __int64 v68; // [rsp+158h] [rbp-148h]
  unsigned __int64 v69; // [rsp+160h] [rbp-140h]
  unsigned __int64 v70; // [rsp+168h] [rbp-138h]
  _QWORD v71[2]; // [rsp+170h] [rbp-130h] BYREF
  unsigned __int64 v72; // [rsp+180h] [rbp-120h]
  unsigned __int64 v73; // [rsp+1D8h] [rbp-C8h]
  _BYTE *v74; // [rsp+1E0h] [rbp-C0h]
  char v75[8]; // [rsp+1F0h] [rbp-B0h] BYREF
  __int64 v76; // [rsp+1F8h] [rbp-A8h]
  unsigned __int64 v77; // [rsp+200h] [rbp-A0h]
  unsigned __int64 v78; // [rsp+258h] [rbp-48h]
  __int64 v79; // [rsp+260h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 224);
  result = *(__int64 **)(v4 + 32);
  v49 = *(__int64 **)(v4 + 40);
  if ( result == v49 )
    return result;
  v50 = *(__int64 **)(v4 + 32);
  do
  {
    v52 = *v50;
    sub_1B1EDD0(v71, &v52);
    v7 = &v55;
    sub_16CCCB0(&v55, (__int64)v61, (__int64)v71);
    v9 = v74;
    v10 = v73;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v11 = (unsigned __int64)&v74[-v73];
    if ( v74 == (_BYTE *)v73 )
    {
      v13 = 0;
    }
    else
    {
      if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_82;
      v12 = sub_22077B0((unsigned __int64)&v74[-v73]);
      v9 = v74;
      v10 = v73;
      v13 = v12;
    }
    v62 = v13;
    v63 = v13;
    v64 = v13 + v11;
    if ( v9 != (_BYTE *)v10 )
    {
      v14 = v13;
      v15 = v10;
      do
      {
        if ( v14 )
        {
          *(_QWORD *)v14 = *(_QWORD *)v15;
          v16 = *(_BYTE *)(v15 + 16);
          *(_BYTE *)(v14 + 16) = v16;
          if ( v16 )
            *(_QWORD *)(v14 + 8) = *(_QWORD *)(v15 + 8);
        }
        v15 += 24LL;
        v14 += 24LL;
      }
      while ( (_BYTE *)v15 != v9 );
      v13 += 8 * ((v15 - 24 - v10) >> 3) + 24;
    }
    v7 = v65;
    v63 = v13;
    v9 = v67;
    sub_16CCCB0(v65, (__int64)v67, (__int64)v75);
    v17 = v79;
    v18 = v78;
    v68 = 0;
    v69 = 0;
    v70 = 0;
    v19 = v79 - v78;
    if ( v79 == v78 )
    {
      v21 = 0;
    }
    else
    {
      if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_82:
        sub_4261EA(v7, v9, v8);
      v20 = sub_22077B0(v79 - v78);
      v17 = v79;
      v18 = v78;
      v21 = v20;
    }
    v68 = v21;
    v22 = v21;
    v69 = v21;
    v70 = v21 + v19;
    if ( v17 != v18 )
    {
      v23 = v18;
      do
      {
        if ( v22 )
        {
          *(_QWORD *)v22 = *(_QWORD *)v23;
          v24 = *(_BYTE *)(v23 + 16);
          *(_BYTE *)(v22 + 16) = v24;
          if ( v24 )
            *(_QWORD *)(v22 + 8) = *(_QWORD *)(v23 + 8);
        }
        v23 += 24LL;
        v22 += 24LL;
      }
      while ( v23 != v17 );
      v22 = v21 + 8 * ((v23 - 24 - v18) >> 3) + 24;
    }
    v25 = v63;
    v26 = v62;
    v69 = v22;
    if ( v63 - v62 == v22 - v21 )
      goto LABEL_45;
    do
    {
LABEL_24:
      v27 = *(_QWORD *)(v25 - 24);
      v28 = sub_16E8750(a2, 2u);
      v29 = sub_1649960(**(_QWORD **)(v27 + 32));
      v31 = *(_WORD **)(v28 + 24);
      v32 = (char *)v29;
      v33 = *(_QWORD *)(v28 + 16) - (_QWORD)v31;
      if ( v30 > v33 )
      {
        v46 = sub_16E7EE0(v28, v32, v30);
        v31 = *(_WORD **)(v46 + 24);
        v28 = v46;
        v33 = *(_QWORD *)(v46 + 16) - (_QWORD)v31;
      }
      else if ( v30 )
      {
        v48 = v30;
        memcpy(v31, v32, v30);
        v47 = *(_QWORD *)(v28 + 16);
        v31 = (_WORD *)(v48 + *(_QWORD *)(v28 + 24));
        *(_QWORD *)(v28 + 24) = v31;
        v33 = v47 - (_QWORD)v31;
      }
      if ( v33 <= 1 )
      {
        sub_16E7EE0(v28, ":\n", 2u);
      }
      else
      {
        *v31 = 2618;
        *(_QWORD *)(v28 + 24) += 2LL;
      }
      v34 = sub_38694E0(a1, v27, a3, a4);
      sub_3862460(v34, a2, 4u);
      v35 = v63;
      do
      {
        v36 = *(_QWORD *)(v35 - 24);
        if ( !*(_BYTE *)(v35 - 8) )
        {
          v37 = *(__int64 **)(v36 + 8);
          *(_BYTE *)(v35 - 8) = 1;
          *(_QWORD *)(v35 - 16) = v37;
          goto LABEL_34;
        }
        while ( 1 )
        {
          v37 = *(__int64 **)(v35 - 16);
LABEL_34:
          if ( *(__int64 **)(v36 + 16) == v37 )
            break;
          *(_QWORD *)(v35 - 16) = v37 + 1;
          v39 = *v37;
          v40 = v56;
          if ( v57 != v56 )
            goto LABEL_32;
          v41 = &v56[v59];
          if ( v56 == v41 )
          {
LABEL_72:
            if ( v59 < v58 )
            {
              ++v59;
              *v41 = v39;
              ++v55;
LABEL_43:
              v53 = v39;
              v54 = 0;
              sub_197E9F0(&v62, (__int64)&v53);
              v25 = v63;
              v26 = v62;
              goto LABEL_44;
            }
LABEL_32:
            sub_16CCBA0((__int64)&v55, v39);
            if ( v38 )
              goto LABEL_43;
          }
          else
          {
            v42 = 0;
            while ( v39 != *v40 )
            {
              if ( *v40 == -2 )
              {
                v42 = v40;
                if ( v40 + 1 == v41 )
                  goto LABEL_42;
                ++v40;
              }
              else if ( v41 == ++v40 )
              {
                if ( !v42 )
                  goto LABEL_72;
LABEL_42:
                *v42 = v39;
                --v60;
                ++v55;
                goto LABEL_43;
              }
            }
          }
        }
        v63 -= 24LL;
        v25 = v62;
        v35 = v63;
      }
      while ( v63 != v62 );
      v26 = v62;
LABEL_44:
      v21 = v68;
    }
    while ( v25 - v26 != v69 - v68 );
LABEL_45:
    if ( v26 != v25 )
    {
      v43 = v21;
      while ( *(_QWORD *)v26 == *(_QWORD *)v43 )
      {
        v44 = *(_BYTE *)(v26 + 16);
        v45 = *(_BYTE *)(v43 + 16);
        if ( v44 && v45 )
        {
          if ( *(_QWORD *)(v26 + 8) != *(_QWORD *)(v43 + 8) )
            goto LABEL_24;
        }
        else if ( v44 != v45 )
        {
          goto LABEL_24;
        }
        v26 += 24LL;
        v43 += 24LL;
        if ( v25 == v26 )
          goto LABEL_52;
      }
      goto LABEL_24;
    }
LABEL_52:
    if ( v21 )
      j_j___libc_free_0(v21);
    if ( v66 != v65[1] )
      _libc_free(v66);
    if ( v62 )
      j_j___libc_free_0(v62);
    if ( v57 != v56 )
      _libc_free((unsigned __int64)v57);
    if ( v78 )
      j_j___libc_free_0(v78);
    if ( v77 != v76 )
      _libc_free(v77);
    if ( v73 )
      j_j___libc_free_0(v73);
    if ( v72 != v71[1] )
      _libc_free(v72);
    result = ++v50;
  }
  while ( v49 != v50 );
  return result;
}
