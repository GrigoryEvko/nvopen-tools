// Function: sub_18B61E0
// Address: 0x18b61e0
//
__int64 *__fastcall sub_18B61E0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        char *a7,
        size_t a8)
{
  __int64 v12; // rax
  size_t v13; // rdx
  _BYTE *v14; // rdi
  char *v15; // rsi
  unsigned __int64 v16; // rax
  void **v17; // r8
  __int64 v18; // rsi
  __int64 *v19; // r13
  void **v20; // rdi
  __int64 v21; // r15
  _BYTE *v22; // rax
  _BYTE *v23; // rax
  void **v24; // r13
  void *v25; // rdi
  __int64 *v26; // rax
  __int64 v28; // rax
  size_t v29; // [rsp+18h] [rbp-98h]
  __int64 v30[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v31[2]; // [rsp+40h] [rbp-70h] BYREF
  void *v32; // [rsp+50h] [rbp-60h] BYREF
  void *v33; // [rsp+58h] [rbp-58h]
  unsigned __int64 v34; // [rsp+60h] [rbp-50h]
  void *dest; // [rsp+68h] [rbp-48h]
  int v36; // [rsp+70h] [rbp-40h]
  __int64 *v37; // [rsp+78h] [rbp-38h]

  v30[0] = (__int64)v31;
  sub_18B4E10(v30, "__typeid_", (__int64)"");
  v36 = 1;
  dest = 0;
  v34 = 0;
  v33 = 0;
  v32 = &unk_49EFBE0;
  v37 = v30;
  v12 = sub_161E970(a2);
  v14 = dest;
  v15 = (char *)v12;
  v16 = v34;
  if ( v13 > v34 - (unsigned __int64)dest )
  {
    v28 = sub_16E7EE0((__int64)&v32, v15, v13);
    v14 = *(_BYTE **)(v28 + 24);
    v17 = (void **)v28;
    v16 = *(_QWORD *)(v28 + 16);
  }
  else
  {
    v17 = &v32;
    if ( v13 )
    {
      v29 = v13;
      memcpy(dest, v15, v13);
      v17 = &v32;
      dest = (char *)dest + v29;
      v14 = dest;
      if ( v34 > (unsigned __int64)dest )
        goto LABEL_4;
      goto LABEL_22;
    }
  }
  if ( v16 > (unsigned __int64)v14 )
  {
LABEL_4:
    v17[3] = v14 + 1;
    *v14 = 95;
    goto LABEL_5;
  }
LABEL_22:
  v17 = (void **)sub_16E7DE0((__int64)v17, 95);
LABEL_5:
  v18 = a3;
  v19 = &a4[a5];
  sub_16E7A90((__int64)v17, v18);
  while ( v19 != a4 )
  {
    v21 = *a4;
    v22 = dest;
    if ( (unsigned __int64)dest < v34 )
    {
      v20 = &v32;
      dest = (char *)dest + 1;
      *v22 = 95;
    }
    else
    {
      v20 = (void **)sub_16E7DE0((__int64)&v32, 95);
    }
    ++a4;
    sub_16E7A90((__int64)v20, v21);
  }
  v23 = dest;
  if ( (unsigned __int64)dest >= v34 )
  {
    v24 = (void **)sub_16E7DE0((__int64)&v32, 95);
  }
  else
  {
    v24 = &v32;
    dest = (char *)dest + 1;
    *v23 = 95;
  }
  v25 = v24[3];
  if ( (_BYTE *)v24[2] - (_BYTE *)v25 < a8 )
  {
    sub_16E7EE0((__int64)v24, a7, a8);
  }
  else if ( a8 )
  {
    memcpy(v25, a7, a8);
    v24[3] = (char *)v24[3] + a8;
  }
  if ( dest != v33 )
    sub_16E7BA0((__int64 *)&v32);
  v26 = v37;
  *a1 = (__int64)(a1 + 2);
  sub_18B4BA0(a1, (_BYTE *)*v26, *v26 + v26[1]);
  sub_16E7BC0((__int64 *)&v32);
  if ( (_QWORD *)v30[0] != v31 )
    j_j___libc_free_0(v30[0], v31[0] + 1LL);
  return a1;
}
