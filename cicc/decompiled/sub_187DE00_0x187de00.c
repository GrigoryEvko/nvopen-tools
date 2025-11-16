// Function: sub_187DE00
// Address: 0x187de00
//
__int64 *__fastcall sub_187DE00(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // r14
  _BYTE *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  _BYTE *v10; // r8
  __int64 v11; // r13
  __int64 v12; // r12
  const void *v13; // r10
  signed __int64 v14; // r12
  const void *v15; // rax
  const void *v16; // rdx
  __int64 *v17; // rsi
  __int64 *result; // rax
  __int64 v19; // rcx
  _BYTE *v20; // r9
  size_t v21; // rsi
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  bool v24; // cf
  unsigned __int64 v25; // rax
  char *v26; // rcx
  const void *v27; // r8
  void *v28; // rcx
  const void *v29; // r9
  size_t v30; // rax
  size_t v31; // r12
  char *v32; // r12
  __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rax
  _BYTE *v36; // rsi
  char *dest; // [rsp+8h] [rbp-98h]
  void *src; // [rsp+10h] [rbp-90h]
  void *v39; // [rsp+18h] [rbp-88h]
  void *v40; // [rsp+18h] [rbp-88h]
  const void *v41; // [rsp+20h] [rbp-80h]
  void *v42; // [rsp+20h] [rbp-80h]
  const void *v43; // [rsp+20h] [rbp-80h]
  char *v44; // [rsp+28h] [rbp-78h]
  const void *v45; // [rsp+28h] [rbp-78h]
  const void *v46; // [rsp+28h] [rbp-78h]
  const void *v47; // [rsp+28h] [rbp-78h]
  unsigned __int64 v48; // [rsp+38h] [rbp-68h]
  char *v49; // [rsp+40h] [rbp-60h]
  _BYTE *v50; // [rsp+40h] [rbp-60h]
  char *v51; // [rsp+48h] [rbp-58h]
  void *v52; // [rsp+48h] [rbp-58h]
  char *v53; // [rsp+48h] [rbp-58h]
  __int64 v54; // [rsp+50h] [rbp-50h]
  __int64 v55; // [rsp+58h] [rbp-48h]
  __int64 v56[7]; // [rsp+68h] [rbp-38h] BYREF

  v3 = a1[1];
  v54 = v3;
  if ( v3 == a1[2] )
  {
    sub_187DC10(a1, (char *)v3);
    v4 = a1[1];
    v54 = v4 - 24;
  }
  else
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = 0;
      *(_QWORD *)(v3 + 8) = 0;
      *(_QWORD *)(v3 + 16) = 0;
      v54 = a1[1];
    }
    v4 = v54 + 24;
    a1[1] = v54 + 24;
  }
  v5 = *(_QWORD *)(a2 + 24);
  v48 = 0xAAAAAAAAAAAAAAABLL * ((v4 - *a1) >> 3) - 1;
  v55 = a2 + 8;
  if ( v5 != a2 + 8 )
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(v5 + 32);
      v8 = a1[3];
      v56[0] = v7;
      v9 = *(_QWORD *)(v8 + 8 * v7);
      if ( !v9 )
      {
        v6 = *(_BYTE **)(v4 - 16);
        if ( v6 == *(_BYTE **)(v4 - 8) )
        {
          sub_9CA200(v54, v6, v56);
        }
        else
        {
          if ( v6 )
          {
            *(_QWORD *)v6 = v7;
            v6 = *(_BYTE **)(v4 - 16);
          }
          *(_QWORD *)(v4 - 16) = v6 + 8;
        }
        goto LABEL_11;
      }
      v10 = *(_BYTE **)(v4 - 16);
      v11 = *a1 + 24 * v9;
      v12 = *(_QWORD *)(v11 + 8);
      v13 = *(const void **)v11;
      if ( *(_QWORD *)v11 != v12 )
        break;
LABEL_11:
      v5 = sub_220EF30(v5);
      if ( v55 == v5 )
        goto LABEL_18;
    }
    v14 = v12 - (_QWORD)v13;
    if ( *(_QWORD *)(v4 - 8) - (_QWORD)v10 >= (unsigned __int64)v14 )
    {
      memmove(*(void **)(v4 - 16), *(const void **)v11, v14);
      *(_QWORD *)(v4 - 16) += v14;
      v15 = *(const void **)v11;
      v16 = *(const void **)(v11 + 8);
      goto LABEL_16;
    }
    v20 = *(_BYTE **)(v4 - 24);
    v21 = v10 - v20;
    v22 = (v10 - v20) >> 3;
    if ( v14 >> 3 > 0xFFFFFFFFFFFFFFFLL - v22 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    v23 = v14 >> 3;
    if ( v14 >> 3 < v22 )
      v23 = (v10 - v20) >> 3;
    v24 = __CFADD__(v22, v23);
    v25 = v22 + v23;
    if ( v24 )
    {
      v34 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v25 )
      {
        v49 = 0;
        v26 = 0;
        goto LABEL_27;
      }
      if ( v25 > 0xFFFFFFFFFFFFFFFLL )
        v25 = 0xFFFFFFFFFFFFFFFLL;
      v34 = 8 * v25;
    }
    v47 = *(const void **)v11;
    v50 = *(_BYTE **)(v4 - 16);
    v53 = (char *)v34;
    v35 = sub_22077B0(v34);
    v10 = v50;
    v20 = *(_BYTE **)(v4 - 24);
    v26 = (char *)v35;
    v13 = v47;
    v36 = v50;
    v49 = &v53[v35];
    v21 = v36 - v20;
LABEL_27:
    v51 = &v26[v21 + v14];
    if ( v10 == v20 )
    {
      v40 = v26;
      v43 = v20;
      v46 = v10;
      memcpy(&v26[v21], v13, v14);
      v27 = v46;
      v29 = v43;
      v28 = v40;
      v30 = *(_QWORD *)(v4 - 16) - (_QWORD)v46;
      if ( *(const void **)(v4 - 16) == v46 )
      {
        v32 = v51;
        if ( !v43 )
          goto LABEL_30;
LABEL_34:
        v33 = *(_QWORD *)(v4 - 8) - (_QWORD)v29;
LABEL_35:
        v52 = v28;
        j_j___libc_free_0(v29, v33);
        v28 = v52;
        goto LABEL_30;
      }
    }
    else
    {
      v41 = v20;
      v39 = v10;
      v44 = v26;
      dest = &v26[v21];
      src = (void *)v13;
      memmove(v26, v20, v21);
      memcpy(dest, src, v14);
      v27 = v39;
      v28 = v44;
      v29 = v41;
      v30 = *(_QWORD *)(v4 - 16) - (_QWORD)v39;
      if ( v39 == *(void **)(v4 - 16) )
      {
        v32 = &v51[v30];
        v33 = *(_QWORD *)(v4 - 8) - (_QWORD)v41;
        goto LABEL_35;
      }
    }
    v42 = v28;
    v45 = v29;
    v31 = v30;
    memcpy(v51, v27, v30);
    v29 = v45;
    v28 = v42;
    v32 = &v51[v31];
    if ( !v45 )
    {
LABEL_30:
      *(_QWORD *)(v4 - 24) = v28;
      *(_QWORD *)(v4 - 16) = v32;
      *(_QWORD *)(v4 - 8) = v49;
      v15 = *(const void **)v11;
      v16 = *(const void **)(v11 + 8);
LABEL_16:
      if ( v16 != v15 )
        *(_QWORD *)(v11 + 8) = v15;
      goto LABEL_11;
    }
    goto LABEL_34;
  }
LABEL_18:
  v17 = *(__int64 **)(v4 - 16);
  for ( result = *(__int64 **)(v4 - 24); result != v17; *(_QWORD *)(a1[3] + 8 * v19) = v48 )
    v19 = *result++;
  return result;
}
