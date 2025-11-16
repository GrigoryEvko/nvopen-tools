// Function: sub_2C3DE40
// Address: 0x2c3de40
//
__int64 __fastcall sub_2C3DE40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rbx
  const void *v9; // r13
  __int64 v10; // r12
  __int64 *v11; // rdi
  __int64 *v12; // rbx
  __int64 v13; // r13
  _QWORD *v14; // rdi
  __int64 v15; // rsi
  _QWORD *v16; // rax
  int v17; // r11d
  _QWORD *v18; // rdi
  __int64 v19; // rsi
  _QWORD *v20; // rax
  int v21; // r11d
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 result; // rax
  __int64 *v26; // [rsp+18h] [rbp-88h]
  __int64 v27; // [rsp+28h] [rbp-78h] BYREF
  __int64 *v28; // [rsp+30h] [rbp-70h] BYREF
  __int64 v29; // [rsp+38h] [rbp-68h]
  _BYTE v30[96]; // [rsp+40h] [rbp-60h] BYREF

  *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
  v8 = *(unsigned int *)(a2 + 64);
  v9 = *(const void **)(a2 + 56);
  v28 = (__int64 *)v30;
  v10 = 8 * v8;
  v29 = 0x600000000LL;
  if ( v8 > 6 )
  {
    sub_C8D5F0((__int64)&v28, v30, v8, 8u, a5, a6);
    v11 = &v28[(unsigned int)v29];
  }
  else
  {
    v11 = (__int64 *)v30;
    if ( !v10 )
      goto LABEL_3;
  }
  memcpy(v11, v9, 8 * v8);
  v11 = v28;
  LODWORD(v10) = v29;
LABEL_3:
  LODWORD(v29) = v10 + v8;
  v26 = &v11[(unsigned int)(v10 + v8)];
  if ( v26 != v11 )
  {
    v12 = v11;
    do
    {
      v13 = *v12;
      v27 = a2;
      v14 = *(_QWORD **)(v13 + 80);
      v15 = (__int64)&v14[*(unsigned int *)(v13 + 88)];
      v16 = sub_2C3D9C0(v14, v15, &v27);
      if ( v16 + 1 != (_QWORD *)v15 )
      {
        memmove(v16, v16 + 1, v15 - (_QWORD)(v16 + 1));
        v17 = *(_DWORD *)(v13 + 88);
      }
      v27 = v13;
      *(_DWORD *)(v13 + 88) = v17 - 1;
      v18 = *(_QWORD **)(a2 + 56);
      v19 = (__int64)&v18[*(unsigned int *)(a2 + 64)];
      v20 = sub_2C3D9C0(v18, v19, &v27);
      a5 = (__int64)(v20 + 1);
      if ( v20 + 1 != (_QWORD *)v19 )
      {
        memmove(v20, v20 + 1, v19 - a5);
        v21 = *(_DWORD *)(a2 + 64);
      }
      *(_DWORD *)(a2 + 64) = v21 - 1;
      v22 = *(unsigned int *)(v13 + 88);
      if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 92) )
      {
        sub_C8D5F0(v13 + 80, (const void *)(v13 + 96), v22 + 1, 8u, a5, a6);
        v22 = *(unsigned int *)(v13 + 88);
      }
      *(_QWORD *)(*(_QWORD *)(v13 + 80) + 8 * v22) = a1;
      ++*(_DWORD *)(v13 + 88);
      v23 = *(unsigned int *)(a1 + 64);
      if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 68) )
      {
        sub_C8D5F0(a1 + 56, (const void *)(a1 + 72), v23 + 1, 8u, a5, a6);
        v23 = *(unsigned int *)(a1 + 64);
      }
      ++v12;
      *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8 * v23) = v13;
      ++*(_DWORD *)(a1 + 64);
    }
    while ( v26 != v12 );
    v11 = v28;
  }
  if ( v11 != (__int64 *)v30 )
    _libc_free((unsigned __int64)v11);
  v24 = *(unsigned int *)(a1 + 88);
  if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
  {
    sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v24 + 1, 8u, a5, a6);
    v24 = *(unsigned int *)(a1 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v24) = a2;
  ++*(_DWORD *)(a1 + 88);
  result = *(unsigned int *)(a2 + 64);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 68) )
  {
    sub_C8D5F0(a2 + 56, (const void *)(a2 + 72), result + 1, 8u, a5, a6);
    result = *(unsigned int *)(a2 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 56) + 8 * result) = a1;
  ++*(_DWORD *)(a2 + 64);
  return result;
}
