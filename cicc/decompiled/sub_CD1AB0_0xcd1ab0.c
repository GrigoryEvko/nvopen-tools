// Function: sub_CD1AB0
// Address: 0xcd1ab0
//
void __fastcall sub_CD1AB0(__int64 a1, __int64 a2, __int16 a3, const void *a4, size_t a5)
{
  char *v8; // r8
  _BYTE *v9; // rcx
  size_t v10; // rdx
  size_t v11; // rax
  __int64 v12; // r15
  size_t v13; // rax
  bool v14; // cf
  __int64 v15; // rax
  __int64 v16; // rbx
  const void *v17; // r10
  char *v18; // rcx
  char *v19; // r11
  char *v20; // rbx
  size_t v21; // rdx
  size_t v22; // r13
  char *v23; // r8
  const void *v24; // r10
  size_t v25; // rax
  unsigned __int64 v26; // rdx
  char *v27; // rdx
  __int64 v28; // rax
  char *v29; // rdx
  char *dest; // [rsp+8h] [rbp-68h]
  const void *v31; // [rsp+10h] [rbp-60h]
  char *v32; // [rsp+10h] [rbp-60h]
  char *v33; // [rsp+18h] [rbp-58h]
  const void *v34; // [rsp+18h] [rbp-58h]
  _BYTE *v35; // [rsp+18h] [rbp-58h]
  const void *v36; // [rsp+18h] [rbp-58h]
  char *src; // [rsp+20h] [rbp-50h]
  _BYTE *srca; // [rsp+20h] [rbp-50h]
  char *srcb; // [rsp+20h] [rbp-50h]
  _BYTE *srcc; // [rsp+20h] [rbp-50h]
  __int64 v41; // [rsp+28h] [rbp-48h]
  char *v42; // [rsp+28h] [rbp-48h]
  unsigned __int8 v43[49]; // [rsp+3Fh] [rbp-31h] BYREF

  sub_CD17A0(a2, a3, *(_DWORD *)(a1 + 8) - *(_DWORD *)a1);
  v8 = *(char **)(a1 + 8);
  v9 = *(_BYTE **)a1;
  v10 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
  v11 = v10;
  if ( a5 )
  {
    v12 = *(_QWORD *)(a1 + 16);
    if ( a5 <= v12 - (__int64)v8 )
    {
      memmove(*(void **)(a1 + 8), a4, a5);
      v9 = *(_BYTE **)a1;
      v8 = (char *)(a5 + *(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = v8;
      v11 = v8 - v9;
      goto LABEL_4;
    }
    if ( a5 > 0x7FFFFFFFFFFFFFFFLL - v10 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    v13 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
    if ( a5 >= v10 )
      v13 = a5;
    v14 = __CFADD__(v10, v13);
    v15 = v10 + v13;
    v16 = v15;
    if ( v14 || v15 < 0 )
    {
      v16 = 0x7FFFFFFFFFFFFFFFLL;
    }
    else if ( !v15 )
    {
      v41 = 0;
      v17 = *(const void **)a1;
      v18 = 0;
      goto LABEL_13;
    }
    v42 = *(char **)(a1 + 8);
    v28 = sub_22077B0(v16);
    v8 = v42;
    v17 = *(const void **)a1;
    v18 = (char *)v28;
    v12 = *(_QWORD *)(a1 + 16);
    v29 = v42;
    v41 = v16 + v28;
    v10 = (size_t)&v29[-*(_QWORD *)a1];
LABEL_13:
    v19 = &v18[v10];
    v20 = &v18[v10 + a5];
    if ( v10 )
    {
      v32 = v8;
      v34 = v17;
      srca = v18;
      dest = &v18[v10];
      memmove(v18, v17, v10);
      memcpy(dest, a4, a5);
      v23 = v32;
      v9 = srca;
      v24 = v34;
      v25 = *(_QWORD *)(a1 + 8) - (_QWORD)v32;
      if ( !v25 )
      {
        v8 = v20;
LABEL_25:
        v35 = v9;
        srcb = v8;
        j_j___libc_free_0(v24, v12 - (_QWORD)v24);
        v9 = v35;
        v8 = srcb;
LABEL_16:
        *(_QWORD *)a1 = v9;
        *(_QWORD *)(a1 + 8) = v8;
        *(_QWORD *)(a1 + 16) = v41;
        v11 = v8 - v9;
        if ( (((_BYTE)v8 - (_BYTE)v9) & 3) == 0 )
          return;
        goto LABEL_17;
      }
    }
    else
    {
      v21 = a5;
      v31 = v17;
      v33 = v18;
      v22 = 0;
      src = v8;
      memcpy(v19, a4, v21);
      v23 = src;
      v9 = v33;
      v24 = v31;
      v25 = *(_QWORD *)(a1 + 8) - (_QWORD)src;
      if ( !v25 )
        goto LABEL_15;
    }
    v36 = v24;
    srcc = v9;
    v22 = v25;
    memcpy(v20, v23, v25);
    v24 = v36;
    v9 = srcc;
LABEL_15:
    v8 = &v20[v22];
    if ( !v24 )
      goto LABEL_16;
    goto LABEL_25;
  }
LABEL_4:
  if ( (v11 & 3) == 0 )
    return;
LABEL_17:
  v43[0] = 0;
  v26 = (v11 & 0xFFFFFFFFFFFFFFFCLL) + 4;
  if ( v26 > v11 )
  {
    sub_CD1880((__int64 *)a1, v8, v26 - v11, v43);
  }
  else if ( v26 < v11 )
  {
    v27 = &v9[v26];
    if ( v8 != v27 )
      *(_QWORD *)(a1 + 8) = v27;
  }
}
