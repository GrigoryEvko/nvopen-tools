// Function: sub_2627E10
// Address: 0x2627e10
//
__int64 *__fastcall sub_2627E10(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // r14
  _BYTE *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  _BYTE *v9; // r8
  __int64 v10; // r13
  __int64 v11; // r12
  const void *v12; // r10
  signed __int64 v13; // r12
  const void *v14; // rax
  const void *v15; // rdx
  __int64 *v16; // rsi
  __int64 *result; // rax
  __int64 v18; // rcx
  _BYTE *v19; // r9
  unsigned __int64 v20; // rdx
  size_t v21; // rsi
  unsigned __int64 v22; // rax
  bool v23; // cf
  unsigned __int64 v24; // rax
  char *v25; // rcx
  const void *v26; // r8
  void *v27; // rcx
  unsigned __int64 v28; // r9
  size_t v29; // rax
  size_t v30; // r12
  char *v31; // r12
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  _BYTE *v34; // rsi
  char *dest; // [rsp+8h] [rbp-98h]
  void *v36; // [rsp+10h] [rbp-90h]
  void *src; // [rsp+18h] [rbp-88h]
  void *srca; // [rsp+18h] [rbp-88h]
  unsigned __int64 v39; // [rsp+20h] [rbp-80h]
  void *v40; // [rsp+20h] [rbp-80h]
  unsigned __int64 v41; // [rsp+20h] [rbp-80h]
  char *v42; // [rsp+28h] [rbp-78h]
  unsigned __int64 v43; // [rsp+28h] [rbp-78h]
  const void *v44; // [rsp+28h] [rbp-78h]
  const void *v45; // [rsp+28h] [rbp-78h]
  unsigned __int64 v46; // [rsp+38h] [rbp-68h]
  char *v47; // [rsp+40h] [rbp-60h]
  _BYTE *v48; // [rsp+40h] [rbp-60h]
  char *v49; // [rsp+48h] [rbp-58h]
  void *v50; // [rsp+48h] [rbp-58h]
  char *v51; // [rsp+48h] [rbp-58h]
  __int64 v52; // [rsp+50h] [rbp-50h]
  __int64 v53; // [rsp+58h] [rbp-48h]
  __int64 v54[7]; // [rsp+68h] [rbp-38h] BYREF

  v2 = a1[1];
  v52 = v2;
  if ( v2 == a1[2] )
  {
    sub_187DC10(a1, (char *)v2);
    v3 = a1[1];
    v52 = v3 - 24;
  }
  else
  {
    if ( v2 )
    {
      *(_QWORD *)v2 = 0;
      *(_QWORD *)(v2 + 8) = 0;
      *(_QWORD *)(v2 + 16) = 0;
      v52 = a1[1];
    }
    v3 = v52 + 24;
    a1[1] = v52 + 24;
  }
  v4 = *(_QWORD *)(a2 + 24);
  v46 = 0xAAAAAAAAAAAAAAABLL * ((v3 - *a1) >> 3) - 1;
  v53 = a2 + 8;
  if ( v4 != a2 + 8 )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v4 + 32);
      v7 = a1[3];
      v54[0] = v6;
      v8 = *(_QWORD *)(v7 + 8 * v6);
      if ( !v8 )
      {
        v5 = *(_BYTE **)(v3 - 16);
        if ( v5 == *(_BYTE **)(v3 - 8) )
        {
          sub_9CA200(v52, v5, v54);
        }
        else
        {
          if ( v5 )
          {
            *(_QWORD *)v5 = v6;
            v5 = *(_BYTE **)(v3 - 16);
          }
          *(_QWORD *)(v3 - 16) = v5 + 8;
        }
        goto LABEL_11;
      }
      v9 = *(_BYTE **)(v3 - 16);
      v10 = *a1 + 24 * v8;
      v11 = *(_QWORD *)(v10 + 8);
      v12 = *(const void **)v10;
      if ( v11 != *(_QWORD *)v10 )
        break;
LABEL_11:
      v4 = sub_220EF30(v4);
      if ( v53 == v4 )
        goto LABEL_18;
    }
    v13 = v11 - (_QWORD)v12;
    if ( *(_QWORD *)(v3 - 8) - (_QWORD)v9 >= (unsigned __int64)v13 )
    {
      memmove(*(void **)(v3 - 16), *(const void **)v10, v13);
      *(_QWORD *)(v3 - 16) += v13;
      v14 = *(const void **)v10;
      v15 = *(const void **)(v10 + 8);
      goto LABEL_16;
    }
    v19 = *(_BYTE **)(v3 - 24);
    v20 = v13 >> 3;
    v21 = v9 - v19;
    v22 = (v9 - v19) >> 3;
    if ( v13 >> 3 > 0xFFFFFFFFFFFFFFFLL - v22 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v20 < v22 )
      v20 = (v9 - v19) >> 3;
    v23 = __CFADD__(v20, v22);
    v24 = v20 + v22;
    if ( v23 )
    {
      v32 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v24 )
      {
        v47 = 0;
        v25 = 0;
        goto LABEL_27;
      }
      if ( v24 > 0xFFFFFFFFFFFFFFFLL )
        v24 = 0xFFFFFFFFFFFFFFFLL;
      v32 = 8 * v24;
    }
    v48 = *(_BYTE **)(v3 - 16);
    v45 = *(const void **)v10;
    v51 = (char *)v32;
    v33 = sub_22077B0(v32);
    v9 = v48;
    v19 = *(_BYTE **)(v3 - 24);
    v25 = (char *)v33;
    v12 = v45;
    v34 = v48;
    v47 = &v51[v33];
    v21 = v34 - v19;
LABEL_27:
    v49 = &v25[v21 + v13];
    if ( v9 == v19 )
    {
      srca = v25;
      v41 = (unsigned __int64)v19;
      v44 = v9;
      memcpy(&v25[v21], v12, v13);
      v26 = v44;
      v28 = v41;
      v27 = srca;
      v29 = *(_QWORD *)(v3 - 16) - (_QWORD)v44;
      if ( *(const void **)(v3 - 16) == v44 )
      {
        v31 = v49;
        if ( !v41 )
          goto LABEL_30;
        goto LABEL_34;
      }
    }
    else
    {
      v39 = (unsigned __int64)v19;
      v36 = v9;
      v42 = v25;
      dest = &v25[v21];
      src = (void *)v12;
      memmove(v25, v19, v21);
      memcpy(dest, src, v13);
      v26 = v36;
      v27 = v42;
      v28 = v39;
      v29 = *(_QWORD *)(v3 - 16) - (_QWORD)v36;
      if ( v36 == *(void **)(v3 - 16) )
      {
        v31 = &v49[v29];
        goto LABEL_34;
      }
    }
    v40 = v27;
    v43 = v28;
    v30 = v29;
    memcpy(v49, v26, v29);
    v28 = v43;
    v27 = v40;
    v31 = &v49[v30];
    if ( !v43 )
    {
LABEL_30:
      *(_QWORD *)(v3 - 24) = v27;
      *(_QWORD *)(v3 - 16) = v31;
      *(_QWORD *)(v3 - 8) = v47;
      v14 = *(const void **)v10;
      v15 = *(const void **)(v10 + 8);
LABEL_16:
      if ( v14 != v15 )
        *(_QWORD *)(v10 + 8) = v14;
      goto LABEL_11;
    }
LABEL_34:
    v50 = v27;
    j_j___libc_free_0(v28);
    v27 = v50;
    goto LABEL_30;
  }
LABEL_18:
  v16 = *(__int64 **)(v3 - 16);
  for ( result = *(__int64 **)(v3 - 24); result != v16; *(_QWORD *)(a1[3] + 8 * v18) = v46 )
    v18 = *result++;
  return result;
}
