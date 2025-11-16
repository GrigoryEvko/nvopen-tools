// Function: sub_C6BF30
// Address: 0xc6bf30
//
__int64 __fastcall sub_C6BF30(__int64 a1, __int64 a2, _QWORD *a3)
{
  int v4; // r13d
  unsigned int v5; // r12d
  __int64 *v7; // r14
  __int64 *v8; // r15
  int v9; // eax
  size_t v10; // rcx
  char *v11; // r10
  int v12; // r11d
  unsigned int i; // r8d
  __int64 v14; // r12
  char *v15; // r13
  size_t v16; // r9
  bool v17; // al
  bool v18; // al
  bool v19; // al
  int v20; // eax
  unsigned int v21; // r8d
  int v22; // eax
  int v23; // eax
  size_t v24; // [rsp+8h] [rbp-108h]
  size_t v25; // [rsp+8h] [rbp-108h]
  size_t v26; // [rsp+10h] [rbp-100h]
  char *v27; // [rsp+10h] [rbp-100h]
  size_t v28; // [rsp+10h] [rbp-100h]
  char *v29; // [rsp+18h] [rbp-F8h]
  unsigned int v30; // [rsp+18h] [rbp-F8h]
  char *v31; // [rsp+18h] [rbp-F8h]
  int v32; // [rsp+20h] [rbp-F0h]
  size_t v33; // [rsp+20h] [rbp-F0h]
  int v34; // [rsp+20h] [rbp-F0h]
  unsigned int v35; // [rsp+28h] [rbp-E8h]
  int v36; // [rsp+28h] [rbp-E8h]
  unsigned int v37; // [rsp+28h] [rbp-E8h]
  int v38; // [rsp+2Ch] [rbp-E4h]
  __int64 v39; // [rsp+30h] [rbp-E0h]
  size_t n; // [rsp+38h] [rbp-D8h]
  __int64 s2; // [rsp+40h] [rbp-D0h]
  size_t v42; // [rsp+48h] [rbp-C8h]
  __int64 v43; // [rsp+50h] [rbp-C0h]
  __int64 v44; // [rsp+58h] [rbp-B8h]
  __int64 *v45; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v46; // [rsp+A8h] [rbp-68h]
  size_t v47; // [rsp+B0h] [rbp-60h]
  __int64 v48[2]; // [rsp+C0h] [rbp-50h] BYREF
  _QWORD v49[8]; // [rsp+D0h] [rbp-40h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v7 = 0;
  v44 = *(_QWORD *)(a1 + 8);
  v42 = 0;
  v43 = -1;
  if ( !(unsigned __int8)sub_C6A630((char *)0xFFFFFFFFFFFFFFFFLL, 0, 0) )
  {
    sub_C6B0E0(v48, -1, 0);
    sub_C6B270(&v45, (__int64)v48);
    v7 = v45;
    v43 = v46;
    v42 = v47;
    if ( (_QWORD *)v48[0] != v49 )
      j_j___libc_free_0(v48[0], v49[0] + 1LL);
  }
  v8 = 0;
  n = 0;
  s2 = -2;
  if ( !(unsigned __int8)sub_C6A630((char *)0xFFFFFFFFFFFFFFFELL, 0, 0) )
  {
    sub_C6B0E0(v48, -2, 0);
    sub_C6B270(&v45, (__int64)v48);
    v8 = v45;
    s2 = v46;
    n = v47;
    if ( (_QWORD *)v48[0] != v49 )
      j_j___libc_free_0(v48[0], v49[0] + 1LL);
  }
  v9 = sub_C94890(*(_QWORD *)(a2 + 8), *(_QWORD *)(a2 + 16));
  v10 = *(_QWORD *)(a2 + 16);
  v11 = *(char **)(a2 + 8);
  v12 = v4 - 1;
  v38 = 1;
  v39 = 0;
  for ( i = (v4 - 1) & v9; ; i = v12 & v21 )
  {
    v14 = v44 + ((unsigned __int64)i << 6);
    v15 = *(char **)(v14 + 8);
    v16 = *(_QWORD *)(v14 + 16);
    v17 = v11 + 1 == 0;
    if ( v15 != (char *)-1LL )
    {
      v17 = v11 + 2 == 0;
      if ( v15 != (char *)-2LL )
      {
        if ( v16 != v10 )
          goto LABEL_10;
        v30 = i;
        v36 = v12;
        v33 = *(_QWORD *)(v14 + 16);
        if ( !v10 )
          goto LABEL_26;
        v24 = v10;
        v27 = v11;
        v22 = memcmp(v11, v15, v10);
        i = v30;
        v12 = v36;
        v16 = v33;
        v10 = v24;
        v11 = v27;
        v17 = v22 == 0;
      }
    }
    if ( v17 )
    {
LABEL_26:
      *a3 = v14;
      v5 = 1;
      goto LABEL_27;
    }
LABEL_10:
    v18 = v15 + 1 == 0;
    if ( v43 == -1 )
      goto LABEL_41;
    v18 = v15 + 2 == 0;
    if ( v43 == -2 )
      goto LABEL_41;
    if ( v16 == v42 )
      break;
LABEL_13:
    v19 = v15 + 1 == 0;
    if ( s2 == -1 )
      goto LABEL_18;
    v19 = v15 + 2 == 0;
    if ( s2 == -2 )
      goto LABEL_18;
    if ( v16 != n )
      goto LABEL_22;
    if ( n )
    {
      v26 = v10;
      v29 = v11;
      v35 = i;
      v32 = v12;
      v20 = memcmp(v15, (const void *)s2, n);
      v10 = v26;
      v11 = v29;
      i = v35;
      v12 = v32;
      v19 = v20 == 0;
LABEL_18:
      if ( !v19 )
        goto LABEL_22;
    }
    if ( v39 )
      v14 = v39;
    v39 = v14;
LABEL_22:
    v21 = v38 + i;
    ++v38;
  }
  v28 = v10;
  v31 = v11;
  v37 = i;
  v34 = v12;
  if ( !v16 )
    goto LABEL_42;
  v25 = v16;
  v23 = memcmp(v15, (const void *)v43, v16);
  v10 = v28;
  v11 = v31;
  i = v37;
  v12 = v34;
  v16 = v25;
  v18 = v23 == 0;
LABEL_41:
  if ( !v18 )
    goto LABEL_13;
LABEL_42:
  if ( v39 )
    v14 = v39;
  *a3 = v14;
  v5 = 0;
LABEL_27:
  if ( v8 )
  {
    if ( (__int64 *)*v8 != v8 + 2 )
      j_j___libc_free_0(*v8, v8[2] + 1);
    j_j___libc_free_0(v8, 32);
  }
  if ( v7 )
  {
    if ( (__int64 *)*v7 != v7 + 2 )
      j_j___libc_free_0(*v7, v7[2] + 1);
    j_j___libc_free_0(v7, 32);
  }
  return v5;
}
