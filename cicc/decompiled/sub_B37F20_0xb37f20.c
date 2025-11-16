// Function: sub_B37F20
// Address: 0xb37f20
//
__int64 __fastcall sub_B37F20(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        unsigned int a6,
        const void *a7,
        __int64 a8)
{
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  _BYTE *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  signed __int64 v18; // rbx
  const void *v19; // r13
  __int64 v20; // r10
  __int64 v21; // rax
  __int64 v22; // rax
  void *v24; // r8
  unsigned __int64 v25; // rax
  signed __int64 v26; // rdx
  unsigned __int64 v27; // rcx
  bool v28; // cf
  unsigned __int64 v29; // rax
  char *v30; // rcx
  void *v31; // rcx
  void *v32; // r8
  __int64 v33; // r10
  size_t v34; // rax
  size_t v35; // rbx
  char *v36; // rbx
  __int64 v37; // rsi
  __int64 v38; // rax
  char *dest; // [rsp+0h] [rbp-80h]
  __int64 v40; // [rsp+8h] [rbp-78h]
  char *v41; // [rsp+8h] [rbp-78h]
  void *v42; // [rsp+10h] [rbp-70h]
  void *v43; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+10h] [rbp-70h]
  char *v45; // [rsp+18h] [rbp-68h]
  __int64 v46; // [rsp+18h] [rbp-68h]
  void *v47; // [rsp+18h] [rbp-68h]
  __int64 v48; // [rsp+20h] [rbp-60h]
  void *src; // [rsp+28h] [rbp-58h]
  char *v51; // [rsp+30h] [rbp-50h]
  void *v52; // [rsp+30h] [rbp-50h]
  __int64 v53; // [rsp+38h] [rbp-48h] BYREF
  _QWORD v54[7]; // [rsp+48h] [rbp-38h] BYREF

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  v12 = *(_QWORD *)(a2 + 72);
  v53 = a5;
  v13 = sub_BCB2E0(v12);
  v54[0] = sub_ACD640(v13, a3, 0);
  sub_B331C0(a1, v54);
  v14 = sub_BCB2D0(*(_QWORD *)(a2 + 72));
  v54[0] = sub_ACD640(v14, a4, 0);
  sub_B331C0(a1, v54);
  v15 = *(_BYTE **)(a1 + 8);
  if ( v15 == *(_BYTE **)(a1 + 16) )
  {
    sub_9281F0(a1, v15, &v53);
  }
  else
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = v53;
      v15 = *(_BYTE **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v15 + 8;
  }
  v16 = sub_BCB2D0(*(_QWORD *)(a2 + 72));
  v54[0] = sub_ACD640(v16, (unsigned int)a8, 0);
  sub_B331C0(a1, v54);
  v17 = sub_BCB2D0(*(_QWORD *)(a2 + 72));
  v54[0] = sub_ACD640(v17, a6, 0);
  sub_B331C0(a1, v54);
  v18 = 8 * a8;
  v19 = *(const void **)(a1 + 8);
  if ( 8 * a8 )
  {
    v20 = *(_QWORD *)(a1 + 16);
    if ( v18 <= (unsigned __int64)(v20 - (_QWORD)v19) )
    {
      memmove(*(void **)(a1 + 8), a7, v18);
      *(_QWORD *)(a1 + 8) += v18;
      goto LABEL_8;
    }
    v24 = *(void **)a1;
    v25 = v18 >> 3;
    v26 = (signed __int64)v19 - *(_QWORD *)a1;
    v27 = v26 >> 3;
    if ( v18 >> 3 > (unsigned __int64)(0xFFFFFFFFFFFFFFFLL - (v26 >> 3)) )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v25 < v27 )
      v25 = ((__int64)v19 - *(_QWORD *)a1) >> 3;
    v28 = __CFADD__(v27, v25);
    v29 = v27 + v25;
    if ( v28 )
    {
      v37 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v29 )
      {
        v48 = 0;
        v30 = 0;
        goto LABEL_15;
      }
      if ( v29 > 0xFFFFFFFFFFFFFFFLL )
        v29 = 0xFFFFFFFFFFFFFFFLL;
      v37 = 8 * v29;
    }
    v38 = sub_22077B0(v37);
    v24 = *(void **)a1;
    v30 = (char *)v38;
    v20 = *(_QWORD *)(a1 + 16);
    v26 = (signed __int64)v19 - *(_QWORD *)a1;
    v48 = v37 + v38;
LABEL_15:
    v51 = &v30[v26 + v18];
    if ( v19 == v24 )
    {
      v41 = v30;
      v44 = v20;
      v35 = 0;
      v47 = v24;
      memcpy(&v30[v26], a7, 8 * a8);
      v32 = v47;
      v33 = v44;
      v31 = v41;
      v34 = *(_QWORD *)(a1 + 8) - (_QWORD)v19;
      if ( *(const void **)(a1 + 8) == v19 )
      {
LABEL_18:
        v36 = &v51[v35];
        if ( !v32 )
        {
LABEL_19:
          *(_QWORD *)a1 = v31;
          *(_QWORD *)(a1 + 8) = v36;
          *(_QWORD *)(a1 + 16) = v48;
          goto LABEL_8;
        }
LABEL_24:
        v52 = v31;
        j_j___libc_free_0(v32, v33 - (_QWORD)v32);
        v31 = v52;
        goto LABEL_19;
      }
    }
    else
    {
      v40 = v20;
      v42 = v24;
      v45 = v30;
      dest = &v30[v26];
      memmove(v30, v24, v26);
      memcpy(dest, a7, v18);
      v31 = v45;
      v32 = v42;
      v33 = v40;
      v34 = *(_QWORD *)(a1 + 8) - (_QWORD)v19;
      if ( v19 == *(const void **)(a1 + 8) )
      {
        v36 = &v51[v34];
        goto LABEL_24;
      }
    }
    v43 = v31;
    v46 = v33;
    v35 = v34;
    src = v32;
    memcpy(v51, v19, v34);
    v31 = v43;
    v33 = v46;
    v32 = src;
    goto LABEL_18;
  }
LABEL_8:
  v21 = sub_BCB2D0(*(_QWORD *)(a2 + 72));
  v54[0] = sub_ACD640(v21, 0, 0);
  sub_B331C0(a1, v54);
  v22 = sub_BCB2D0(*(_QWORD *)(a2 + 72));
  v54[0] = sub_ACD640(v22, 0, 0);
  sub_B331C0(a1, v54);
  return a1;
}
