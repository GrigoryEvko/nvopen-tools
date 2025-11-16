// Function: sub_9D2B60
// Address: 0x9d2b60
//
unsigned __int64 *__fastcall sub_9D2B60(unsigned __int64 *a1, __int64 a2)
{
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 *v8; // r14
  unsigned __int64 *v9; // rdi
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  char v12; // dl
  char v13; // al
  size_t v14; // rdx
  __int64 *v15; // rdx
  bool v16; // zf
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // [rsp+0h] [rbp-250h] BYREF
  __int64 v24; // [rsp+8h] [rbp-248h]
  __int64 v25; // [rsp+10h] [rbp-240h] BYREF
  __int64 v26; // [rsp+18h] [rbp-238h]
  unsigned __int64 v27; // [rsp+20h] [rbp-230h] BYREF
  size_t n; // [rsp+28h] [rbp-228h]
  _QWORD src[2]; // [rsp+30h] [rbp-220h] BYREF
  char v30; // [rsp+40h] [rbp-210h]
  char v31; // [rsp+210h] [rbp-40h]

  if ( **(_BYTE **)a2 )
  {
    *a1 = 1;
    return a1;
  }
  **(_BYTE **)a2 = 1;
  sub_A87B50(
    &v27,
    **(_QWORD **)(a2 + 8),
    *(_QWORD *)(*(_QWORD *)(a2 + 8) + 8LL),
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 440LL) + 232LL),
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 440LL) + 240LL));
  v8 = *(unsigned __int64 **)(a2 + 8);
  v9 = (unsigned __int64 *)*v8;
  if ( (_QWORD *)v27 == src )
  {
    v14 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v9 = src[0];
      else
        memcpy(v9, src, n);
      v14 = n;
      v9 = (unsigned __int64 *)*v8;
    }
    v8[1] = v14;
    *((_BYTE *)v9 + v14) = 0;
    v9 = (unsigned __int64 *)v27;
    goto LABEL_8;
  }
  if ( v9 == v8 + 2 )
  {
    *v8 = v27;
    v8[1] = n;
    v8[2] = src[0];
    goto LABEL_27;
  }
  *v8 = v27;
  v10 = v8[2];
  v8[1] = n;
  v8[2] = src[0];
  if ( !v9 )
  {
LABEL_27:
    v27 = (unsigned __int64)src;
    v9 = src;
    goto LABEL_8;
  }
  v27 = (unsigned __int64)v9;
  src[0] = v10;
LABEL_8:
  n = 0;
  *(_BYTE *)v9 = 0;
  if ( (_QWORD *)v27 != src )
    j_j___libc_free_0(v27, src[0] + 1LL);
  v11 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)(v11 + 32) )
  {
    v15 = *(__int64 **)(a2 + 8);
    v16 = *(_QWORD *)(v11 + 16) == 0;
    v17 = *v15;
    v18 = v15[1];
    v19 = *(_QWORD *)(a2 + 16);
    v25 = v17;
    v20 = *(_QWORD *)(v19 + 440);
    v26 = v18;
    v21 = *(_QWORD *)(v20 + 240);
    v22 = *(_QWORD *)(v20 + 232);
    v24 = v21;
    v23 = v22;
    if ( v16 )
      sub_4263D6(v21, v17, v22);
    (*(void (__fastcall **)(unsigned __int64 *, __int64, __int64 *, __int64 *))(v11 + 24))(&v27, v11, &v23, &v25);
    if ( v30 )
    {
      sub_2240AE0(*(_QWORD *)(a2 + 8), &v27);
      if ( v30 )
      {
        v30 = 0;
        if ( (_QWORD *)v27 != src )
          j_j___libc_free_0(v27, src[0] + 1LL);
      }
    }
  }
  sub_AE41B0(&v27, **(_QWORD **)(a2 + 8), *(_QWORD *)(*(_QWORD *)(a2 + 8) + 8LL), v5, v6, v7, v23, v24, v25, v26);
  v12 = v31 & 1;
  v31 = (2 * (v31 & 1)) | v31 & 0xFD;
  if ( v12 )
  {
    *a1 = v27 | 1;
  }
  else
  {
    sub_BA9570(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 440LL), &v27);
    v13 = v31;
    *a1 = 1;
    if ( (v13 & 2) != 0 )
      sub_9D2AF0(&v27);
    if ( (v13 & 1) != 0 )
    {
      if ( v27 )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v27 + 8LL))(v27);
    }
    else
    {
      sub_AE4030(&v27);
    }
  }
  return a1;
}
