// Function: sub_2BE23C0
// Address: 0x2be23c0
//
__int64 __fastcall sub_2BE23C0(__int64 a1, unsigned int a2)
{
  __int64 v3; // rdi
  const __m128i **v4; // rdx
  __int64 v5; // r13
  size_t v6; // rdx
  unsigned __int64 v7; // r13
  __int64 *v8; // rbx
  __int64 *v9; // r14
  unsigned __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // r14
  unsigned __int64 v14; // rdi
  __int64 v15; // r12
  __int64 v16; // rbx
  unsigned __int64 v17; // rdi
  unsigned __int64 i; // r12
  unsigned __int64 v20; // rdi
  unsigned __int8 v21; // [rsp+7h] [rbp-49h]
  _QWORD v22[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a1 + 96;
  v4 = *(const __m128i ***)(v3 - 32);
  v22[0] = *(_QWORD *)(v3 + 32);
  sub_2BE0D90(v3, v22, v4);
  *(_BYTE *)(a1 + 140) = 0;
  v5 = *(_QWORD *)(a1 + 96);
  if ( *(_QWORD *)(a1 + 104) != v5 )
  {
    v21 = 0;
    while ( 1 )
    {
      v6 = 0xAAAAAAAAAAAAAAABLL
         * ((__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 64LL) - *(_QWORD *)(*(_QWORD *)(a1 + 56) + 56LL)) >> 4);
      if ( v6 )
        memset(*(void **)(a1 + 120), 0, v6);
      v7 = *(_QWORD *)(a1 + 96);
      *(_QWORD *)(a1 + 96) = 0;
      v8 = *(__int64 **)(a1 + 104);
      *(_QWORD *)(a1 + 112) = 0;
      v9 = (__int64 *)v7;
      *(_QWORD *)(a1 + 104) = 0;
      while ( v8 != v9 )
      {
        v10 = *(_QWORD *)a1;
        *(_QWORD *)a1 = v9[1];
        *(_QWORD *)(a1 + 8) = v9[2];
        *(_QWORD *)(a1 + 16) = v9[3];
        v9[1] = 0;
        v9[2] = 0;
        v9[3] = 0;
        if ( v10 )
          j_j___libc_free_0(v10);
        v11 = *v9;
        v9 += 4;
        sub_2BE1DD0(a1, a2, v11);
      }
      if ( (_BYTE)a2 == 1 )
        v21 |= *(_BYTE *)(a1 + 140);
      v12 = *(_QWORD *)(a1 + 24);
      if ( v12 == *(_QWORD *)(a1 + 40) )
        break;
      v13 = v7;
      for ( *(_QWORD *)(a1 + 24) = v12 + 1; v8 != (__int64 *)v13; v13 += 32LL )
      {
        v14 = *(_QWORD *)(v13 + 8);
        if ( v14 )
          j_j___libc_free_0(v14);
      }
      if ( v7 )
        j_j___libc_free_0(v7);
      *(_BYTE *)(a1 + 140) = 0;
      v5 = *(_QWORD *)(a1 + 96);
      if ( *(_QWORD *)(a1 + 104) == v5 )
      {
        v15 = *(_QWORD *)(a1 + 96);
        goto LABEL_20;
      }
    }
    for ( i = v7; v8 != (__int64 *)i; i += 32LL )
    {
      v20 = *(_QWORD *)(i + 8);
      if ( v20 )
        j_j___libc_free_0(v20);
    }
    if ( v7 )
      j_j___libc_free_0(v7);
    v5 = *(_QWORD *)(a1 + 96);
    v15 = *(_QWORD *)(a1 + 104);
    if ( (_BYTE)a2 )
      goto LABEL_22;
    goto LABEL_21;
  }
  v21 = 0;
  v15 = *(_QWORD *)(a1 + 96);
LABEL_20:
  if ( !(_BYTE)a2 )
  {
LABEL_21:
    v21 = *(_BYTE *)(a1 + 140);
LABEL_22:
    if ( v5 != v15 )
    {
      v16 = v5;
      do
      {
        v17 = *(_QWORD *)(v16 + 8);
        if ( v17 )
          j_j___libc_free_0(v17);
        v16 += 32;
      }
      while ( v16 != v15 );
      *(_QWORD *)(a1 + 104) = v5;
    }
  }
  return v21;
}
