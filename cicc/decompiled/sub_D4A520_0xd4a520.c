// Function: sub_D4A520
// Address: 0xd4a520
//
__m128i *__fastcall sub_D4A520(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  _QWORD *v12; // rdx
  _BYTE *v13; // r15
  unsigned __int8 v14; // al
  _BYTE **v15; // rax
  _BYTE *v16; // r12
  __int64 v17; // r14
  __int64 v18; // rax
  size_t v19; // rbx
  const void *v20; // r13
  const void *v21; // rax
  size_t v22; // rdx
  size_t v23; // rbx
  const void *v24; // r13
  const void *v25; // rax
  size_t v26; // rdx
  size_t v27; // rbx
  const void *v28; // r13
  const void *v29; // rax
  size_t v30; // rdx
  size_t v31; // rbx
  const void *v32; // r13
  const void *v33; // rax
  size_t v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rdx
  unsigned __int64 v38; // rax
  __int64 v39; // rbx
  __int64 *v40; // rsi
  __int64 *v41; // rcx
  __int64 v42; // rax
  __m128i *v43; // r12
  size_t v45; // rbx
  const void *v46; // r13
  const void *v47; // rax
  size_t v48; // rdx
  size_t v49; // rbx
  const void *v50; // r13
  const void *v51; // rax
  size_t v52; // rdx
  size_t v53; // rbx
  const void *v54; // r13
  const void *v55; // rax
  size_t v56; // rdx
  __int64 v60; // [rsp+18h] [rbp-A8h]
  __int64 v61; // [rsp+20h] [rbp-A0h]
  __int64 v63; // [rsp+38h] [rbp-88h]
  _QWORD *v64; // [rsp+40h] [rbp-80h]
  __int64 v65; // [rsp+48h] [rbp-78h]
  _QWORD *v66; // [rsp+50h] [rbp-70h]
  __int64 v67; // [rsp+58h] [rbp-68h]
  __int64 *v68; // [rsp+60h] [rbp-60h] BYREF
  __int64 v69; // [rsp+68h] [rbp-58h]
  _QWORD v70[10]; // [rsp+70h] [rbp-50h] BYREF

  v68 = v70;
  v69 = 0x400000001LL;
  v70[0] = 0;
  if ( a2 )
  {
    v6 = *(_BYTE *)(a2 - 16);
    if ( (v6 & 2) != 0 )
    {
      v8 = *(_QWORD *)(a2 - 32);
      v7 = *(unsigned int *)(a2 - 24);
    }
    else
    {
      v7 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
      v8 = a2 - 8LL * ((v6 >> 2) & 0xF) - 16;
    }
    v9 = sub_D46550(v8, v7, 1);
    v64 = v12;
    if ( (_QWORD *)v9 != v12 )
    {
      v66 = (_QWORD *)v9;
      v61 = 16 * v10;
      v60 = (16 * v10) >> 6;
      v65 = a3 + ((16 * v10) & 0xFFFFFFFFFFFFFFC0LL);
      while ( 1 )
      {
        v13 = (_BYTE *)*v66;
        if ( (unsigned __int8)(*(_BYTE *)*v66 - 5) > 0x1Fu )
          goto LABEL_30;
        v14 = *(v13 - 16);
        v15 = (v14 & 2) != 0 ? (_BYTE **)*((_QWORD *)v13 - 4) : (_BYTE **)&v13[-8 * ((v14 >> 2) & 0xF) - 16];
        v16 = *v15;
        if ( **v15 )
          goto LABEL_30;
        v17 = a3;
        v63 = v61 + a3;
        v18 = v61 >> 4;
        if ( v60 > 0 )
        {
          while ( 1 )
          {
            v19 = *(_QWORD *)(v17 + 8);
            v20 = *(const void **)v17;
            v21 = (const void *)sub_B91420((__int64)v16);
            if ( v19 <= v22 && (!v19 || !memcmp(v21, v20, v19)) )
              goto LABEL_8;
            v23 = *(_QWORD *)(v17 + 24);
            v24 = *(const void **)(v17 + 16);
            v67 = v17 + 16;
            v25 = (const void *)sub_B91420((__int64)v16);
            if ( v23 <= v26 && (!v23 || !memcmp(v25, v24, v23)) )
              goto LABEL_45;
            v27 = *(_QWORD *)(v17 + 40);
            v28 = *(const void **)(v17 + 32);
            v67 = v17 + 32;
            v29 = (const void *)sub_B91420((__int64)v16);
            if ( v27 <= v30 && (!v27 || !memcmp(v29, v28, v27)) )
              goto LABEL_45;
            v31 = *(_QWORD *)(v17 + 56);
            v32 = *(const void **)(v17 + 48);
            v67 = v17 + 48;
            v33 = (const void *)sub_B91420((__int64)v16);
            if ( v31 <= v34 && (!v31 || !memcmp(v33, v32, v31)) )
            {
LABEL_45:
              v17 = v67;
              goto LABEL_8;
            }
            v17 += 64;
            if ( v65 == v17 )
            {
              v18 = (v63 - v65) >> 4;
              break;
            }
          }
        }
        if ( v18 == 2 )
          goto LABEL_49;
        if ( v18 != 3 )
        {
          if ( v18 != 1 )
            goto LABEL_30;
          goto LABEL_53;
        }
        v45 = *(_QWORD *)(v17 + 8);
        v46 = *(const void **)v17;
        v47 = (const void *)sub_B91420((__int64)v16);
        if ( v45 > v48 || v45 && memcmp(v47, v46, v45) )
          break;
LABEL_8:
        if ( v63 == v17 )
        {
LABEL_30:
          v35 = (unsigned int)v69;
          v36 = (unsigned int)v69 + 1LL;
          if ( v36 > HIDWORD(v69) )
          {
            sub_C8D5F0((__int64)&v68, v70, v36, 8u, v11, a6);
            v35 = (unsigned int)v69;
          }
          ++v66;
          v68[v35] = (__int64)v13;
          LODWORD(v69) = v69 + 1;
          if ( v64 == v66 )
          {
LABEL_33:
            v37 = (unsigned int)v69;
            v38 = HIDWORD(v69);
            goto LABEL_35;
          }
        }
        else if ( v64 == ++v66 )
        {
          goto LABEL_33;
        }
      }
      v17 += 16;
LABEL_49:
      v49 = *(_QWORD *)(v17 + 8);
      v50 = *(const void **)v17;
      v51 = (const void *)sub_B91420((__int64)v16);
      if ( v49 > v52 || v49 && memcmp(v51, v50, v49) )
      {
        v17 += 16;
LABEL_53:
        v53 = *(_QWORD *)(v17 + 8);
        v54 = *(const void **)v17;
        v55 = (const void *)sub_B91420((__int64)v16);
        if ( v53 > v56 || v53 && memcmp(v55, v54, v53) )
          goto LABEL_30;
        goto LABEL_8;
      }
      goto LABEL_8;
    }
  }
  v38 = 4;
  v37 = 1;
LABEL_35:
  v39 = (8 * a6) >> 3;
  if ( v38 < v37 + v39 )
  {
    sub_C8D5F0((__int64)&v68, v70, v37 + v39, 8u, v37 + v39, a6);
    v37 = (unsigned int)v69;
  }
  v40 = v68;
  v41 = &v68[v37];
  if ( 8 * a6 > 0 )
  {
    v42 = 0;
    do
    {
      v41[v42] = *(_QWORD *)(a5 + 8 * v42);
      ++v42;
    }
    while ( v39 - v42 > 0 );
    v40 = v68;
    LODWORD(v37) = v69;
  }
  LODWORD(v69) = v39 + v37;
  v43 = (__m128i *)sub_B9C770(a1, v40, (__int64 *)(unsigned int)(v39 + v37), 1, 1);
  sub_BA6610(v43, 0, (unsigned __int8 *)v43);
  if ( v68 != v70 )
    _libc_free(v68, 0);
  return v43;
}
