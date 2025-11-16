// Function: sub_B2A430
// Address: 0xb2a430
//
__int64 __fastcall sub_B2A430(__int64 a1, unsigned __int64 *a2, __int64 a3)
{
  __int64 v3; // rsi
  char *v4; // r12
  char *v5; // rbx
  char *v6; // rdi
  char *v7; // rdi
  __int64 *v8; // r12
  __int64 *v9; // rbx
  __int64 result; // rax
  __int64 *v11; // rdi
  __int64 *v12; // rdi
  _BYTE v13[8]; // [rsp+0h] [rbp-2E0h] BYREF
  char v14; // [rsp+8h] [rbp-2D8h]
  __int64 *v15; // [rsp+10h] [rbp-2D0h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-2C8h]
  char v17; // [rsp+130h] [rbp-1B0h] BYREF
  char v18; // [rsp+138h] [rbp-1A8h]
  char *v19; // [rsp+140h] [rbp-1A0h] BYREF
  unsigned int v20; // [rsp+148h] [rbp-198h]
  char v21; // [rsp+260h] [rbp-80h] BYREF
  char *v22; // [rsp+268h] [rbp-78h]
  char v23; // [rsp+278h] [rbp-68h] BYREF

  sub_B26B80((__int64)v13, a2, a3, 1u);
  v3 = (__int64)v13;
  sub_B2A420(a1, (__int64)v13, 0);
  if ( v22 != &v23 )
    _libc_free(v22, v13);
  if ( (v18 & 1) != 0 )
  {
    v5 = &v21;
    v4 = (char *)&v19;
  }
  else
  {
    v4 = v19;
    v3 = 72LL * v20;
    if ( !v20 || (v5 = &v19[v3], &v19[v3] == v19) )
    {
LABEL_28:
      sub_C7D6A0(v4, v3, 8);
      if ( (v14 & 1) == 0 )
        goto LABEL_15;
LABEL_29:
      v9 = (__int64 *)&v17;
      v8 = (__int64 *)&v15;
      goto LABEL_17;
    }
  }
  do
  {
    if ( *(_QWORD *)v4 != -4096 && *(_QWORD *)v4 != -8192 )
    {
      v6 = (char *)*((_QWORD *)v4 + 5);
      if ( v6 != v4 + 56 )
        _libc_free(v6, v3);
      v7 = (char *)*((_QWORD *)v4 + 1);
      if ( v7 != v4 + 24 )
        _libc_free(v7, v3);
    }
    v4 += 72;
  }
  while ( v4 != v5 );
  if ( (v18 & 1) == 0 )
  {
    v4 = v19;
    v3 = 72LL * v20;
    goto LABEL_28;
  }
  if ( (v14 & 1) != 0 )
    goto LABEL_29;
LABEL_15:
  v8 = v15;
  v3 = 72LL * v16;
  if ( !v16 )
    return sub_C7D6A0(v8, v3, 8);
  v9 = (__int64 *)((char *)v15 + v3);
  if ( (__int64 *)((char *)v15 + v3) == v15 )
    return sub_C7D6A0(v8, v3, 8);
  do
  {
LABEL_17:
    result = *v8;
    if ( *v8 != -4096 && result != -8192 )
    {
      v11 = (__int64 *)v8[5];
      if ( v11 != v8 + 7 )
        _libc_free(v11, v3);
      v12 = (__int64 *)v8[1];
      result = (__int64)(v8 + 3);
      if ( v12 != v8 + 3 )
        result = _libc_free(v12, v3);
    }
    v8 += 9;
  }
  while ( v8 != v9 );
  if ( (v14 & 1) == 0 )
  {
    v8 = v15;
    v3 = 72LL * v16;
    return sub_C7D6A0(v8, v3, 8);
  }
  return result;
}
