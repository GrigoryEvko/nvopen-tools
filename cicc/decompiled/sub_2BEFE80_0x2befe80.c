// Function: sub_2BEFE80
// Address: 0x2befe80
//
void __fastcall sub_2BEFE80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // r12
  const void *v9; // r8
  __int64 v10; // r13
  __int64 *v11; // rbx
  _QWORD *v12; // rdi
  __int64 v13; // r13
  __int64 v14; // rsi
  _QWORD *v15; // rax
  int v16; // r11d
  _QWORD *v17; // rdi
  __int64 v18; // rsi
  _QWORD *v19; // rax
  int v20; // r11d
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 *v25; // rdi
  __int64 *v26; // rdi
  __int64 *i; // [rsp+18h] [rbp-88h]
  const void *v28; // [rsp+18h] [rbp-88h]
  __int64 v29; // [rsp+28h] [rbp-78h] BYREF
  __int64 *v30; // [rsp+30h] [rbp-70h] BYREF
  __int64 v31; // [rsp+38h] [rbp-68h]
  _BYTE v32[96]; // [rsp+40h] [rbp-60h] BYREF

  *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
  v8 = *(unsigned int *)(a2 + 88);
  v9 = *(const void **)(a2 + 80);
  v30 = (__int64 *)v32;
  v10 = 8 * v8;
  v31 = 0x600000000LL;
  if ( v8 > 6 )
  {
    v28 = v9;
    sub_C8D5F0((__int64)&v30, v32, v8, 8u, (__int64)v9, a6);
    v9 = v28;
    v26 = &v30[(unsigned int)v31];
  }
  else
  {
    v11 = (__int64 *)v32;
    if ( !v10 )
      goto LABEL_3;
    v26 = (__int64 *)v32;
  }
  memcpy(v26, v9, 8 * v8);
  v11 = v30;
  LODWORD(v10) = v31;
LABEL_3:
  LODWORD(v31) = v10 + v8;
  for ( i = &v11[(unsigned int)(v10 + v8)]; i != v11; ++*(_DWORD *)(v13 + 64) )
  {
    v12 = *(_QWORD **)(a2 + 80);
    v13 = *v11;
    v14 = (__int64)&v12[*(unsigned int *)(a2 + 88)];
    v29 = *v11;
    v15 = sub_2BEF2F0(v12, v14, &v29);
    if ( v15 + 1 != (_QWORD *)v14 )
    {
      memmove(v15, v15 + 1, v14 - (_QWORD)(v15 + 1));
      v16 = *(_DWORD *)(a2 + 88);
    }
    v29 = a2;
    *(_DWORD *)(a2 + 88) = v16 - 1;
    v17 = *(_QWORD **)(v13 + 56);
    v18 = (__int64)&v17[*(unsigned int *)(v13 + 64)];
    v19 = sub_2BEF2F0(v17, v18, &v29);
    v9 = v19 + 1;
    if ( v19 + 1 != (_QWORD *)v18 )
    {
      memmove(v19, v19 + 1, v18 - (_QWORD)v9);
      v20 = *(_DWORD *)(v13 + 64);
    }
    *(_DWORD *)(v13 + 64) = v20 - 1;
    v21 = *(unsigned int *)(a1 + 88);
    if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
    {
      sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v21 + 1, 8u, (__int64)v9, a6);
      v21 = *(unsigned int *)(a1 + 88);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v21) = v13;
    ++*(_DWORD *)(a1 + 88);
    v22 = *(unsigned int *)(v13 + 64);
    if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 68) )
    {
      sub_C8D5F0(v13 + 56, (const void *)(v13 + 72), v22 + 1, 8u, (__int64)v9, a6);
      v22 = *(unsigned int *)(v13 + 64);
    }
    ++v11;
    *(_QWORD *)(*(_QWORD *)(v13 + 56) + 8 * v22) = a1;
  }
  v23 = *(unsigned int *)(a2 + 88);
  if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 92) )
  {
    sub_C8D5F0(a2 + 80, (const void *)(a2 + 96), v23 + 1, 8u, (__int64)v9, a6);
    v23 = *(unsigned int *)(a2 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 80) + 8 * v23) = a1;
  ++*(_DWORD *)(a2 + 88);
  v24 = *(unsigned int *)(a1 + 64);
  if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 68) )
  {
    sub_C8D5F0(a1 + 56, (const void *)(a1 + 72), v24 + 1, 8u, (__int64)v9, a6);
    v24 = *(unsigned int *)(a1 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8 * v24) = a2;
  v25 = v30;
  ++*(_DWORD *)(a1 + 64);
  if ( v25 != (__int64 *)v32 )
    _libc_free((unsigned __int64)v25);
}
