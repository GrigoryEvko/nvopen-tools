// Function: sub_2C75640
// Address: 0x2c75640
//
__int64 *__fastcall sub_2C75640(__int64 *a1, __int64 a2)
{
  _BYTE *v4; // rax
  unsigned __int8 v5; // dl
  _BYTE **v6; // rax
  char *v7; // rsi
  __int64 v8; // rdx
  unsigned __int8 v9; // dl
  __int64 *v10; // rax
  size_t v11; // r8
  size_t v12; // rax
  void *v13; // r9
  _BYTE *v14; // rdi
  __int64 *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r9
  unsigned int v18; // r13d
  size_t v19; // rdi
  size_t v20; // r13
  const void *v21; // r8
  _BYTE *v22; // rsi
  __int64 p_src; // rdi
  __int64 v25; // rdi
  _BYTE *v26; // rax
  size_t v27; // [rsp+0h] [rbp-140h]
  size_t v28; // [rsp+8h] [rbp-138h]
  void *v29; // [rsp+8h] [rbp-138h]
  const void *v30; // [rsp+8h] [rbp-138h]
  unsigned __int64 v31[2]; // [rsp+10h] [rbp-130h] BYREF
  _BYTE v32[16]; // [rsp+20h] [rbp-120h] BYREF
  void *src; // [rsp+30h] [rbp-110h] BYREF
  size_t n; // [rsp+38h] [rbp-108h]
  __int64 v35; // [rsp+40h] [rbp-100h] BYREF
  _BYTE *v36; // [rsp+48h] [rbp-F8h]
  _BYTE *v37; // [rsp+50h] [rbp-F0h]
  __int64 v38; // [rsp+58h] [rbp-E8h]
  unsigned __int64 *v39; // [rsp+60h] [rbp-E0h]
  _BYTE *v40; // [rsp+70h] [rbp-D0h] BYREF
  size_t v41; // [rsp+78h] [rbp-C8h]
  unsigned __int64 v42; // [rsp+80h] [rbp-C0h]
  _BYTE v43[184]; // [rsp+88h] [rbp-B8h] BYREF

  v40 = v43;
  v41 = 0;
  v42 = 128;
  v4 = (_BYTE *)sub_B92180(a2);
  if ( !v4 )
    goto LABEL_29;
  if ( *v4 != 16 )
  {
    v5 = *(v4 - 16);
    v6 = (v5 & 2) != 0 ? (_BYTE **)*((_QWORD *)v4 - 4) : (_BYTE **)&v4[-8 * ((v5 >> 2) & 0xF) - 16];
    v4 = *v6;
    if ( !v4 )
    {
      v7 = (char *)byte_3F871B3;
      v8 = 0;
      goto LABEL_11;
    }
  }
  v9 = *(v4 - 16);
  v10 = (v9 & 2) != 0 ? (__int64 *)*((_QWORD *)v4 - 4) : (__int64 *)&v4[-8 * ((v9 >> 2) & 0xF) - 16];
  if ( !*v10 || (v7 = (char *)sub_B91420(*v10)) == 0 )
  {
LABEL_29:
    LOBYTE(v35) = 0;
    src = &v35;
    n = 0;
    goto LABEL_16;
  }
LABEL_11:
  src = &v35;
  sub_2C75590((__int64 *)&src, v7, (__int64)&v7[v8]);
  v11 = n;
  v12 = v41;
  v13 = src;
  if ( n + v41 > v42 )
  {
    v27 = n;
    v29 = src;
    sub_C8D290((__int64)&v40, v43, n + v41, 1u, n, (__int64)src);
    v12 = v41;
    v11 = v27;
    v13 = v29;
    v14 = &v40[v41];
    if ( v27 )
      goto LABEL_13;
  }
  else
  {
    v14 = &v40[v41];
    if ( n )
    {
LABEL_13:
      v28 = v11;
      memcpy(v14, v13, v11);
      v15 = (__int64 *)src;
      v12 = v28 + v41;
      goto LABEL_14;
    }
  }
  v15 = (__int64 *)src;
LABEL_14:
  v41 = v12;
  if ( v15 != &v35 )
    j_j___libc_free_0((unsigned __int64)v15);
LABEL_16:
  v38 = 0x100000000LL;
  v31[0] = (unsigned __int64)v32;
  v31[1] = 0;
  src = &unk_49DD210;
  v32[0] = 0;
  n = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v39 = v31;
  sub_CB5980((__int64)&src, 0, 0, 0);
  v16 = sub_B92180(a2);
  if ( v16 && (v18 = *(_DWORD *)(v16 + 16)) != 0 )
  {
    if ( v36 == v37 )
    {
      p_src = sub_CB6200((__int64)&src, (unsigned __int8 *)"(", 1u);
    }
    else
    {
      *v37 = 40;
      p_src = (__int64)&src;
      ++v37;
    }
    v25 = sub_CB59D0(p_src, v18);
    v26 = *(_BYTE **)(v25 + 32);
    if ( *(_BYTE **)(v25 + 24) == v26 )
    {
      sub_CB6200(v25, (unsigned __int8 *)")", 1u);
    }
    else
    {
      *v26 = 41;
      ++*(_QWORD *)(v25 + 32);
    }
  }
  else if ( (unsigned __int64)(v36 - v37) <= 1 )
  {
    sub_CB6200((__int64)&src, (unsigned __int8 *)"()", 2u);
  }
  else
  {
    *(_WORD *)v37 = 10536;
    v37 += 2;
  }
  v19 = v41;
  v20 = v39[1];
  v21 = (const void *)*v39;
  if ( v20 + v41 > v42 )
  {
    v30 = (const void *)*v39;
    sub_C8D290((__int64)&v40, v43, v20 + v41, 1u, (__int64)v21, v17);
    v19 = v41;
    v21 = v30;
  }
  v22 = v40;
  if ( v20 )
  {
    memcpy(&v40[v19], v21, v20);
    v22 = v40;
    v19 = v41;
  }
  *a1 = (__int64)(a1 + 2);
  v41 = v19 + v20;
  sub_2C75590(a1, v22, (__int64)&v22[v19 + v20]);
  src = &unk_49DD210;
  sub_CB5840((__int64)&src);
  if ( (_BYTE *)v31[0] != v32 )
    j_j___libc_free_0(v31[0]);
  if ( v40 != v43 )
    _libc_free((unsigned __int64)v40);
  return a1;
}
