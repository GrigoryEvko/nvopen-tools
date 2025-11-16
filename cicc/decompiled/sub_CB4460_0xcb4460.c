// Function: sub_CB4460
// Address: 0xcb4460
//
_QWORD *__fastcall sub_CB4460(__int64 a1, unsigned __int64 p_src)
{
  __int64 v2; // r13
  __int64 v3; // rdx
  unsigned __int64 v4; // rcx
  _QWORD *v5; // r15
  __int64 v6; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  char *v10; // rax
  size_t v11; // rdx
  size_t v12; // r12
  char *v13; // r14
  __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  size_t v16; // r14
  char *v17; // r12
  __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  __int64 v20; // rdx
  unsigned __int64 v21; // rcx
  _QWORD *v22; // r15
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  _QWORD *v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r14
  char *v33; // rax
  size_t v34; // rdx
  size_t v35; // r15
  char *v36; // r12
  int v37; // eax
  int v38; // eax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r14
  int v42; // eax
  unsigned int v43; // r9d
  _QWORD *v44; // r10
  _QWORD *v45; // rax
  __int64 v46; // rax
  unsigned int v47; // r9d
  _QWORD *v48; // r10
  _QWORD *v49; // rcx
  _QWORD *v50; // rdx
  _QWORD *v51; // rdx
  void *v52; // r9
  __int64 v53; // rbx
  _QWORD *v54; // rax
  _QWORD *v55; // r13
  void *v56; // r15
  const void *v57; // r15
  __int64 v58; // rax
  _QWORD *v59; // [rsp+0h] [rbp-180h]
  _QWORD *v60; // [rsp+10h] [rbp-170h]
  unsigned int v61; // [rsp+1Ch] [rbp-164h]
  __int64 v62; // [rsp+28h] [rbp-158h]
  __int64 v63; // [rsp+28h] [rbp-158h]
  _QWORD *v64; // [rsp+30h] [rbp-150h]
  __int64 v65; // [rsp+30h] [rbp-150h]
  __int64 *v66; // [rsp+38h] [rbp-148h]
  void *v67; // [rsp+38h] [rbp-148h]
  __int64 v68; // [rsp+40h] [rbp-140h]
  _QWORD v69[4]; // [rsp+50h] [rbp-130h] BYREF
  __int16 v70; // [rsp+70h] [rbp-110h]
  _QWORD v71[4]; // [rsp+80h] [rbp-100h] BYREF
  __int16 v72; // [rsp+A0h] [rbp-E0h]
  void *src; // [rsp+B0h] [rbp-D0h] BYREF
  size_t n; // [rsp+B8h] [rbp-C8h]
  __int64 v75; // [rsp+C0h] [rbp-C0h]
  _BYTE v76[184]; // [rsp+C8h] [rbp-B8h] BYREF

  v2 = p_src;
  src = v76;
  n = 0;
  v75 = 128;
  switch ( *(_DWORD *)(p_src + 32) )
  {
    case 0:
      v8 = *(_QWORD *)(a1 + 208);
      *(_QWORD *)(a1 + 288) += 8LL;
      v9 = ((v8 + 7) & 0xFFFFFFFFFFFFFFF8LL) + 8;
      if ( *(_QWORD *)(a1 + 216) >= v9 && v8 )
      {
        *(_QWORD *)(a1 + 208) = v9;
        v5 = (_QWORD *)((v8 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      }
      else
      {
        p_src = 8;
        v5 = (_QWORD *)sub_9D1E70(a1 + 208, 8, 8, 3);
      }
      *v5 = v2;
      goto LABEL_14;
    case 1:
      p_src = (unsigned __int64)&src;
      v10 = sub_CA8C30(v2, &src);
      v12 = n;
      if ( n )
      {
        v13 = *(char **)(a1 + 112);
        *(_QWORD *)(a1 + 192) += n;
        v56 = src;
        if ( *(_QWORD *)(a1 + 120) >= (unsigned __int64)&v13[v12] && v13 )
          *(_QWORD *)(a1 + 112) = &v13[v12];
        else
          v13 = (char *)sub_9D1E70(a1 + 112, v12, v12, 0);
        p_src = (unsigned __int64)v56;
        memmove(v13, v56, v12);
      }
      else
      {
        v13 = v10;
        v12 = v11;
      }
      v14 = *(_QWORD *)(a1 + 304);
      *(_QWORD *)(a1 + 384) += 24LL;
      v15 = ((v14 + 7) & 0xFFFFFFFFFFFFFFF8LL) + 24;
      if ( *(_QWORD *)(a1 + 312) >= v15 && v14 )
      {
        *(_QWORD *)(a1 + 304) = v15;
        v5 = (_QWORD *)((v14 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      }
      else
      {
        p_src = 24;
        v5 = (_QWORD *)sub_9D1E70(a1 + 304, 24, 24, 3);
      }
      *v5 = v2;
      v5[1] = v13;
      v5[2] = v12;
      goto LABEL_14;
    case 2:
      v16 = *(_QWORD *)(p_src + 80);
      v17 = 0;
      if ( v16 )
      {
        v17 = *(char **)(a1 + 112);
        v57 = *(const void **)(p_src + 72);
        *(_QWORD *)(a1 + 192) += v16;
        if ( *(_QWORD *)(a1 + 120) >= (unsigned __int64)&v17[v16] && v17 )
          *(_QWORD *)(a1 + 112) = &v17[v16];
        else
          v17 = (char *)sub_9D1E70(a1 + 112, v16, v16, 0);
        p_src = (unsigned __int64)v57;
        memmove(v17, v57, v16);
      }
      v18 = *(_QWORD *)(a1 + 304);
      *(_QWORD *)(a1 + 384) += 24LL;
      v19 = ((v18 + 7) & 0xFFFFFFFFFFFFFFF8LL) + 24;
      if ( *(_QWORD *)(a1 + 312) >= v19 && v18 )
      {
        *(_QWORD *)(a1 + 304) = v19;
        v5 = (_QWORD *)((v18 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      }
      else
      {
        p_src = 24;
        v5 = (_QWORD *)sub_9D1E70(a1 + 304, 24, 24, 3);
      }
      *v5 = v2;
      v5[1] = v17;
      v5[2] = v16;
      goto LABEL_14;
    case 4:
      v20 = *(_QWORD *)(a1 + 400);
      *(_QWORD *)(a1 + 480) += 240LL;
      v21 = ((v20 + 7) & 0xFFFFFFFFFFFFFFF8LL) + 240;
      if ( *(_QWORD *)(a1 + 408) >= v21 && v20 )
      {
        *(_QWORD *)(a1 + 400) = v21;
        v22 = (_QWORD *)((v20 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      }
      else
      {
        p_src = 240;
        v22 = (_QWORD *)sub_9D1E70(a1 + 400, 240, 240, 3);
      }
      *v22 = v2;
      v22[3] = 0x2000000000LL;
      v22[4] = v22 + 6;
      v22[5] = 0x600000000LL;
      v22[1] = 0;
      v22[2] = 0;
      *(_BYTE *)(v2 + 76) = 0;
      sub_CAEB90(v2, p_src);
      if ( !*(_QWORD *)(v2 + 80) )
        v2 = 0;
      v68 = v2;
      v26 = v22;
      break;
    case 5:
      v3 = *(_QWORD *)(a1 + 496);
      *(_QWORD *)(a1 + 576) += 32LL;
      v4 = ((v3 + 7) & 0xFFFFFFFFFFFFFFF8LL) + 32;
      if ( *(_QWORD *)(a1 + 504) >= v4 && v3 )
      {
        *(_QWORD *)(a1 + 496) = v4;
        v5 = (_QWORD *)((v3 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      }
      else
      {
        p_src = 32;
        v5 = (_QWORD *)sub_9D1E70(a1 + 496, 32, 32, 3);
      }
      *v5 = v2;
      v5[1] = 0;
      v5[2] = 0;
      v5[3] = 0;
      *(_BYTE *)(v2 + 76) = 0;
      sub_CAEEC0(v2, p_src);
      if ( !*(_QWORD *)(v2 + 80) )
        v2 = 0;
      do
      {
        if ( !v2 )
          break;
        v6 = sub_CB4460(a1, *(_QWORD *)(v2 + 80));
        p_src = *(unsigned int *)(a1 + 96);
        v71[0] = v6;
        if ( (_DWORD)p_src )
          break;
        p_src = v5[2];
        if ( p_src == v5[3] )
        {
          sub_CB4150((__int64)(v5 + 1), (_BYTE *)p_src, v71);
        }
        else
        {
          if ( p_src )
          {
            *(_QWORD *)p_src = v6;
            p_src = v5[2];
          }
          p_src += 8LL;
          v5[2] = p_src;
        }
        sub_CAEEC0(v2, p_src);
      }
      while ( *(_QWORD *)(v2 + 80) );
      goto LABEL_14;
    default:
      v5 = 0;
      v71[0] = "unknown node kind";
      v72 = 259;
      sub_CB1010(a1, p_src, (__int64)v71);
      goto LABEL_14;
  }
  while ( 1 )
  {
    if ( !v68 )
    {
LABEL_68:
      v5 = v26;
      goto LABEL_14;
    }
    v27 = *(_QWORD *)(v68 + 80);
    v28 = sub_CAE820(v27, p_src, v23, v24, v25);
    v32 = v28;
    if ( !v28 || *(_DWORD *)(v28 + 32) != 1 )
      break;
    v64 = sub_CAE940(v27, p_src, v29, v30, v31);
    if ( !v64 )
    {
      v5 = v26;
      v53 = a1;
      goto LABEL_71;
    }
    n = 0;
    v33 = sub_CA8C30(v32, &src);
    v35 = n;
    if ( n )
    {
      v36 = *(char **)(a1 + 112);
      *(_QWORD *)(a1 + 192) += n;
      v52 = src;
      if ( *(_QWORD *)(a1 + 120) >= (unsigned __int64)&v36[v35] && v36 )
      {
        *(_QWORD *)(a1 + 112) = &v36[v35];
      }
      else
      {
        v67 = src;
        v58 = sub_9D1E70(a1 + 112, v35, v35, 0);
        v52 = v67;
        v36 = (char *)v58;
      }
      memmove(v36, v52, v35);
    }
    else
    {
      v36 = v33;
      v35 = v34;
    }
    v66 = v26 + 1;
    v62 = v26[1] + 8LL * *((unsigned int *)v26 + 4);
    v37 = sub_C92610();
    v38 = sub_C92860(v26 + 1, v36, v35, v37);
    if ( v38 == -1 )
      v39 = v26[1] + 8LL * *((unsigned int *)v26 + 4);
    else
      v39 = v26[1] + 8LL * v38;
    if ( v39 != v62 )
    {
      v70 = 1283;
      v69[0] = "duplicated mapping key '";
      v71[0] = v69;
      v69[2] = v36;
      v69[3] = v35;
      v71[2] = "'";
      v72 = 770;
      sub_CB1010(a1, v32, (__int64)v71);
    }
    p_src = (unsigned __int64)v64;
    v65 = sub_CB4460(a1, v64);
    if ( *(_DWORD *)(a1 + 96) )
      goto LABEL_68;
    v40 = *(_QWORD *)(v32 + 16);
    v41 = *(_QWORD *)(v32 + 24);
    v63 = v40;
    v42 = sub_C92610();
    p_src = (unsigned __int64)v36;
    v43 = sub_C92740((__int64)v66, v36, v35, v42);
    v44 = (_QWORD *)(v26[1] + 8LL * v43);
    v45 = (_QWORD *)*v44;
    if ( *v44 )
    {
      if ( v45 != (_QWORD *)-8LL )
        goto LABEL_51;
      --*((_DWORD *)v26 + 6);
    }
    v60 = v44;
    v61 = v43;
    v46 = sub_C7D670(v35 + 33, 8);
    v47 = v61;
    v48 = v60;
    v49 = (_QWORD *)v46;
    if ( v35 )
    {
      v59 = (_QWORD *)v46;
      memcpy((void *)(v46 + 32), v36, v35);
      v47 = v61;
      v48 = v60;
      v49 = v59;
    }
    *((_BYTE *)v49 + v35 + 32) = 0;
    p_src = v47;
    *v49 = v35;
    v49[1] = 0;
    v49[2] = 0;
    v49[3] = 0;
    *v48 = v49;
    ++*((_DWORD *)v26 + 5);
    v50 = (_QWORD *)(v26[1] + 8LL * (unsigned int)sub_C929D0(v66, v47));
    v45 = (_QWORD *)*v50;
    if ( *v50 == -8 || !v45 )
    {
      v51 = v50 + 1;
      do
      {
        do
          v45 = (_QWORD *)*v51++;
        while ( !v45 );
      }
      while ( v45 == (_QWORD *)-8LL );
    }
LABEL_51:
    v45[3] = v41;
    v45[1] = v65;
    v45[2] = v63;
    sub_CAEB90(v68, p_src);
    if ( !*(_QWORD *)(v68 + 80) )
      v68 = 0;
  }
  v5 = v26;
  v53 = a1;
  v54 = sub_CAE940(v27, p_src, v29, v30, v31);
  p_src = v32;
  v55 = v54;
  v71[0] = "Map key must be a scalar";
  v72 = 259;
  sub_CB1010(a1, v32, (__int64)v71);
  if ( v55 )
    goto LABEL_14;
LABEL_71:
  p_src = v32;
  v71[0] = "Map value must not be empty";
  v72 = 259;
  sub_CB1010(v53, v32, (__int64)v71);
LABEL_14:
  if ( src != v76 )
    _libc_free(src, p_src);
  return v5;
}
