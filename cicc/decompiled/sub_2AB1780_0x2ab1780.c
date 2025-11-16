// Function: sub_2AB1780
// Address: 0x2ab1780
//
__int64 __fastcall sub_2AB1780(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r15
  const void *v4; // r9
  __int64 v5; // r8
  _BYTE *v6; // rdi
  __int64 *v7; // r8
  __int64 *v8; // r9
  __int64 v9; // rax
  __int64 *v10; // r10
  unsigned __int64 v11; // r15
  __int64 *v12; // r9
  __int64 v13; // r8
  _BYTE *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r10
  __int64 *v17; // r13
  __int64 *v18; // r15
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 *v21; // r13
  __int64 result; // rax
  __int64 *v23; // r15
  __int64 v24; // r14
  __int64 *v25; // [rsp+8h] [rbp-88h]
  const void *v26; // [rsp+8h] [rbp-88h]
  __int64 v27; // [rsp+18h] [rbp-78h] BYREF
  _BYTE *v28; // [rsp+20h] [rbp-70h] BYREF
  __int64 v29; // [rsp+28h] [rbp-68h]
  _BYTE v30[96]; // [rsp+30h] [rbp-60h] BYREF

  v3 = *(unsigned int *)(a1 + 64);
  v4 = *(const void **)(a1 + 56);
  v28 = v30;
  v29 = 0x600000000LL;
  v5 = 8 * v3;
  if ( v3 > 6 )
  {
    v26 = v4;
    sub_C8D5F0((__int64)&v28, v30, v3, 8u, v5, (__int64)v4);
    v4 = v26;
    v5 = 8 * v3;
    v6 = &v28[8 * (unsigned int)v29];
  }
  else
  {
    v6 = v30;
    if ( !v5 )
      goto LABEL_3;
  }
  memcpy(v6, v4, v5);
  v6 = v28;
  LODWORD(v5) = v29;
LABEL_3:
  LODWORD(v29) = v5 + v3;
  if ( &v6[8 * (unsigned int)(v5 + v3)] != v6 )
  {
    v7 = (__int64 *)v6;
    v8 = &v27;
    do
    {
      v9 = *v7;
      v27 = a1;
      *sub_2AA88F0(*(_QWORD **)(v9 + 80), *(_QWORD *)(v9 + 80) + 8LL * *(unsigned int *)(v9 + 88), v8) = a2;
    }
    while ( v10 != v7 );
    v6 = v28;
  }
  if ( v6 != v30 )
    _libc_free((unsigned __int64)v6);
  v11 = *(unsigned int *)(a1 + 88);
  v12 = *(__int64 **)(a1 + 80);
  v28 = v30;
  v29 = 0x600000000LL;
  v13 = 8 * v11;
  if ( v11 > 6 )
  {
    v25 = v12;
    sub_C8D5F0((__int64)&v28, v30, v11, 8u, v13, (__int64)v12);
    v12 = v25;
    v13 = 8 * v11;
    v14 = &v28[8 * (unsigned int)v29];
  }
  else
  {
    v14 = v30;
    if ( !v13 )
      goto LABEL_11;
  }
  memcpy(v14, v12, v13);
  v14 = v28;
  v13 = (unsigned int)v29;
LABEL_11:
  LODWORD(v29) = v13 + v11;
  if ( &v14[8 * (unsigned int)(v13 + v11)] != v14 )
  {
    v13 = (__int64)v14;
    v12 = &v27;
    do
    {
      v15 = *(_QWORD *)v13;
      v27 = a1;
      *sub_2AA88F0(*(_QWORD **)(v15 + 56), *(_QWORD *)(v15 + 56) + 8LL * *(unsigned int *)(v15 + 64), v12) = a2;
    }
    while ( v16 != v13 );
    v14 = v28;
  }
  if ( v14 != v30 )
    _libc_free((unsigned __int64)v14);
  v17 = *(__int64 **)(a1 + 56);
  v18 = &v17[*(unsigned int *)(a1 + 64)];
  if ( v17 != v18 )
  {
    v19 = *(unsigned int *)(a2 + 64);
    do
    {
      v20 = *v17;
      if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 68) )
      {
        sub_C8D5F0(a2 + 56, (const void *)(a2 + 72), v19 + 1, 8u, v13, (__int64)v12);
        v19 = *(unsigned int *)(a2 + 64);
      }
      ++v17;
      *(_QWORD *)(*(_QWORD *)(a2 + 56) + 8 * v19) = v20;
      v19 = (unsigned int)(*(_DWORD *)(a2 + 64) + 1);
      *(_DWORD *)(a2 + 64) = v19;
    }
    while ( v18 != v17 );
  }
  v21 = *(__int64 **)(a1 + 80);
  result = *(unsigned int *)(a1 + 88);
  v23 = &v21[result];
  if ( v21 != v23 )
  {
    result = *(unsigned int *)(a2 + 88);
    do
    {
      v24 = *v21;
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 92) )
      {
        sub_C8D5F0(a2 + 80, (const void *)(a2 + 96), result + 1, 8u, v13, (__int64)v12);
        result = *(unsigned int *)(a2 + 88);
      }
      ++v21;
      *(_QWORD *)(*(_QWORD *)(a2 + 80) + 8 * result) = v24;
      result = (unsigned int)(*(_DWORD *)(a2 + 88) + 1);
      *(_DWORD *)(a2 + 88) = result;
    }
    while ( v23 != v21 );
  }
  *(_DWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  return result;
}
