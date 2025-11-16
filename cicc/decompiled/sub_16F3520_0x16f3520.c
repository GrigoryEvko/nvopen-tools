// Function: sub_16F3520
// Address: 0x16f3520
//
__int64 __fastcall sub_16F3520(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, int a5)
{
  int v6; // r13d
  unsigned int v7; // r12d
  unsigned __int8 **v9; // r14
  __int64 v10; // rcx
  int v11; // r8d
  unsigned __int8 **v12; // r15
  int v13; // r13d
  int v14; // eax
  size_t v15; // r10
  char *v16; // r8
  unsigned int i; // ecx
  __int64 v18; // r12
  char *v19; // r9
  __int64 v20; // r11
  bool v21; // al
  bool v22; // al
  bool v23; // al
  unsigned int v24; // ecx
  int v25; // eax
  int v26; // eax
  size_t v27; // [rsp+0h] [rbp-110h]
  char *v28; // [rsp+0h] [rbp-110h]
  char *v29; // [rsp+8h] [rbp-108h]
  char *v30; // [rsp+8h] [rbp-108h]
  char *v31; // [rsp+10h] [rbp-100h]
  size_t v32; // [rsp+10h] [rbp-100h]
  char *v33; // [rsp+10h] [rbp-100h]
  unsigned int v34; // [rsp+18h] [rbp-F8h]
  unsigned int v35; // [rsp+18h] [rbp-F8h]
  size_t v36; // [rsp+18h] [rbp-F8h]
  __int64 v37; // [rsp+20h] [rbp-F0h]
  __int64 v38; // [rsp+20h] [rbp-F0h]
  unsigned int v39; // [rsp+20h] [rbp-F0h]
  int v40; // [rsp+2Ch] [rbp-E4h]
  __int64 v41; // [rsp+30h] [rbp-E0h]
  size_t v42; // [rsp+38h] [rbp-D8h]
  __int64 v43; // [rsp+40h] [rbp-D0h]
  size_t n; // [rsp+48h] [rbp-C8h]
  __int64 s2; // [rsp+50h] [rbp-C0h]
  __int64 v46; // [rsp+58h] [rbp-B8h]
  unsigned __int8 **v47; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v48; // [rsp+A8h] [rbp-68h]
  size_t v49; // [rsp+B0h] [rbp-60h]
  __int64 v50[2]; // [rsp+C0h] [rbp-50h] BYREF
  _QWORD v51[8]; // [rsp+D0h] [rbp-40h] BYREF

  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    *a3 = 0;
    return 0;
  }
  v9 = 0;
  v46 = *(_QWORD *)(a1 + 8);
  n = 0;
  s2 = -1;
  if ( !(unsigned __int8)sub_16F23B0((unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0, 0, a4, a5) )
  {
    sub_16F2420(v50, (unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0);
    sub_16F25B0(&v47, (__int64)v50);
    v9 = v47;
    s2 = v48;
    n = v49;
    if ( (_QWORD *)v50[0] != v51 )
      j_j___libc_free_0(v50[0], v51[0] + 1LL);
  }
  v12 = 0;
  v42 = 0;
  v43 = -2;
  if ( !(unsigned __int8)sub_16F23B0((unsigned __int8 *)0xFFFFFFFFFFFFFFFELL, 0, 0, v10, v11) )
  {
    sub_16F2420(v50, (unsigned __int8 *)0xFFFFFFFFFFFFFFFELL, 0);
    sub_16F25B0(&v47, (__int64)v50);
    v12 = v47;
    v43 = v48;
    v42 = v49;
    if ( (_QWORD *)v50[0] != v51 )
      j_j___libc_free_0(v50[0], v51[0] + 1LL);
  }
  v13 = v6 - 1;
  v14 = sub_16D3930(*(_QWORD **)(a2 + 8), *(_QWORD *)(a2 + 16));
  v15 = *(_QWORD *)(a2 + 16);
  v16 = *(char **)(a2 + 8);
  v40 = 1;
  v41 = 0;
  for ( i = v13 & v14; ; i = v13 & v24 )
  {
    v18 = v46 + ((unsigned __int64)i << 6);
    v19 = *(char **)(v18 + 8);
    v20 = *(_QWORD *)(v18 + 16);
    v21 = v16 + 1 == 0;
    if ( v19 == (char *)-1LL || (v21 = v16 + 2 == 0, v19 == (char *)-2LL) )
    {
      if ( v21 )
        goto LABEL_18;
    }
    else if ( v20 == v15 )
    {
      v34 = i;
      v37 = *(_QWORD *)(v18 + 16);
      if ( !v15 )
        goto LABEL_18;
      v27 = v15;
      v29 = *(char **)(v18 + 8);
      v31 = v16;
      v25 = memcmp(v16, v19, v15);
      v16 = v31;
      v19 = v29;
      v15 = v27;
      v20 = v37;
      i = v34;
      if ( !v25 )
      {
LABEL_18:
        *a3 = v18;
        v7 = 1;
        goto LABEL_19;
      }
    }
    v22 = v19 + 1 == 0;
    if ( s2 != -1 )
    {
      v22 = v19 + 2 == 0;
      if ( s2 != -2 )
        break;
    }
    if ( v22 )
      goto LABEL_35;
LABEL_13:
    v23 = v19 + 1 == 0;
    if ( v43 == -1 || (v23 = v19 + 2 == 0, v43 == -2) )
    {
      if ( !v23 )
        goto LABEL_16;
    }
    else
    {
      if ( v20 != v42 )
        goto LABEL_16;
      if ( v42 )
      {
        v33 = v16;
        v36 = v15;
        v39 = i;
        v26 = memcmp(v19, (const void *)v43, v42);
        i = v39;
        v15 = v36;
        v16 = v33;
        if ( v26 )
          goto LABEL_16;
      }
    }
    if ( v41 )
      v18 = v41;
    v41 = v18;
LABEL_16:
    v24 = v40 + i;
    ++v40;
  }
  if ( n != v20 )
    goto LABEL_13;
  v30 = v16;
  v32 = v15;
  v35 = i;
  v38 = v20;
  if ( n )
  {
    v28 = v19;
    if ( memcmp(v19, (const void *)s2, n) )
    {
      v19 = v28;
      v20 = v38;
      i = v35;
      v15 = v32;
      v16 = v30;
      goto LABEL_13;
    }
  }
LABEL_35:
  if ( v41 )
    v18 = v41;
  *a3 = v18;
  v7 = 0;
LABEL_19:
  if ( v12 )
  {
    if ( *v12 != (unsigned __int8 *)(v12 + 2) )
      j_j___libc_free_0(*v12, v12[2] + 1);
    j_j___libc_free_0(v12, 32);
  }
  if ( v9 )
  {
    if ( *v9 != (unsigned __int8 *)(v9 + 2) )
      j_j___libc_free_0(*v9, v9[2] + 1);
    j_j___libc_free_0(v9, 32);
  }
  return v7;
}
