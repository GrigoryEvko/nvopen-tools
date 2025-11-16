// Function: sub_31C3EB0
// Address: 0x31c3eb0
//
__int64 __fastcall sub_31C3EB0(__int64 a1, const __m128i *a2)
{
  __int64 v4; // rdx
  int v5; // ecx
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // rax
  unsigned int v10; // esi
  int v11; // eax
  _BYTE *v12; // r13
  int v13; // eax
  __int64 v14; // r8
  __int64 v15; // rdx
  __m128i v16; // xmm0
  unsigned __int64 v17; // rdi
  __int64 v18; // rdx
  unsigned __int64 v19; // rsi
  int v20; // eax
  __int64 v21; // rdi
  char *v22; // rsi
  __int64 v23; // rdi
  _BYTE *v24; // r12
  _BYTE *v25; // r14
  __int64 v26; // rdi
  _BYTE *v27; // r15
  _BYTE *v28; // r12
  __int64 v29; // rdi
  unsigned __int64 v30; // r12
  __int64 v31; // rdi
  int v32; // [rsp+10h] [rbp-F0h] BYREF
  __int64 v33; // [rsp+18h] [rbp-E8h]
  __int64 v34; // [rsp+20h] [rbp-E0h]
  int v35; // [rsp+28h] [rbp-D8h]
  _BYTE *v36; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v37; // [rsp+38h] [rbp-C8h]
  _BYTE v38[48]; // [rsp+40h] [rbp-C0h] BYREF
  __m128i v39; // [rsp+70h] [rbp-90h] BYREF
  __int64 v40; // [rsp+80h] [rbp-80h]
  _BYTE *v41; // [rsp+88h] [rbp-78h]
  __int64 v42; // [rsp+90h] [rbp-70h]
  _BYTE v43[104]; // [rsp+98h] [rbp-68h] BYREF

  v4 = a2->m128i_i64[1];
  v5 = a2->m128i_i32[0];
  v35 = 0;
  v6 = a2[1].m128i_i64[0];
  v33 = v4;
  v32 = v5;
  v34 = v6;
  if ( (unsigned __int8)sub_31C3810(a1, &v32, &v36) )
  {
    v8 = *((unsigned int *)v36 + 6);
    return *(_QWORD *)(a1 + 32) + 88 * v8 + 24;
  }
  v10 = *(_DWORD *)(a1 + 24);
  v11 = *(_DWORD *)(a1 + 16);
  v12 = v36;
  ++*(_QWORD *)a1;
  v13 = v11 + 1;
  v14 = 2 * v10;
  v39.m128i_i64[0] = (__int64)v12;
  if ( 4 * v13 >= 3 * v10 )
  {
    sub_31C3CB0(a1, v14);
  }
  else
  {
    if ( v10 - *(_DWORD *)(a1 + 20) - v13 > v10 >> 3 )
      goto LABEL_6;
    sub_31C3CB0(a1, v10);
  }
  sub_31C3810(a1, &v32, &v39);
  v12 = (_BYTE *)v39.m128i_i64[0];
  v13 = *(_DWORD *)(a1 + 16) + 1;
LABEL_6:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *((_QWORD *)v12 + 2) != -4096 || *((_QWORD *)v12 + 1) != -4096 || *(_DWORD *)v12 != 0x7FFFFFFF )
    --*(_DWORD *)(a1 + 20);
  *((_QWORD *)v12 + 2) = v34;
  *((_QWORD *)v12 + 1) = v33;
  *(_DWORD *)v12 = v32;
  *((_DWORD *)v12 + 6) = v35;
  v15 = a2[1].m128i_i64[0];
  v16 = _mm_loadu_si128(a2);
  v17 = *(unsigned int *)(a1 + 44);
  v41 = v43;
  v40 = v15;
  v18 = *(unsigned int *)(a1 + 40);
  v36 = v38;
  v19 = v18 + 1;
  v37 = 0x600000000LL;
  v42 = 0x600000000LL;
  v20 = v18;
  v39 = v16;
  if ( v18 + 1 > v17 )
  {
    v30 = *(_QWORD *)(a1 + 32);
    v31 = a1 + 32;
    if ( v30 > (unsigned __int64)&v39 || (unsigned __int64)&v39 >= v30 + 88 * v18 )
    {
      sub_31C3960(v31, v19, v18, (__int64)&v39, v14, v7);
      v18 = *(unsigned int *)(a1 + 40);
      v21 = *(_QWORD *)(a1 + 32);
      v20 = *(_DWORD *)(a1 + 40);
      v22 = (char *)&v39;
    }
    else
    {
      sub_31C3960(v31, v19, v18, (__int64)&v39, v14, v7);
      v21 = *(_QWORD *)(a1 + 32);
      v18 = *(unsigned int *)(a1 + 40);
      v20 = *(_DWORD *)(a1 + 40);
      v22 = &v39.m128i_i8[v21 - v30];
    }
  }
  else
  {
    v21 = *(_QWORD *)(a1 + 32);
    v22 = (char *)&v39;
  }
  v23 = v21 + 88 * v18;
  if ( v23 )
  {
    *(_DWORD *)v23 = *(_DWORD *)v22;
    *(_QWORD *)(v23 + 8) = *((_QWORD *)v22 + 1);
    *(_QWORD *)(v23 + 16) = *((_QWORD *)v22 + 2);
    *(_QWORD *)(v23 + 24) = v23 + 40;
    *(_QWORD *)(v23 + 32) = 0x600000000LL;
    if ( *((_DWORD *)v22 + 8) )
    {
      v22 += 24;
      sub_31C3510(v23 + 24, (__int64)v22);
    }
    v20 = *(_DWORD *)(a1 + 40);
  }
  v24 = v41;
  *(_DWORD *)(a1 + 40) = v20 + 1;
  v25 = &v24[8 * (unsigned int)v42];
  if ( v24 != v25 )
  {
    do
    {
      v26 = *((_QWORD *)v25 - 1);
      v25 -= 8;
      if ( v26 )
        (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)v26 + 8LL))(v26, v22);
    }
    while ( v24 != v25 );
    v25 = v41;
  }
  if ( v25 != v43 )
    _libc_free((unsigned __int64)v25);
  v27 = v36;
  v28 = &v36[8 * (unsigned int)v37];
  if ( v36 != v28 )
  {
    do
    {
      v29 = *((_QWORD *)v28 - 1);
      v28 -= 8;
      if ( v29 )
        (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)v29 + 8LL))(v29, v22);
    }
    while ( v27 != v28 );
    v28 = v36;
  }
  if ( v28 != v38 )
    _libc_free((unsigned __int64)v28);
  v8 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *((_DWORD *)v12 + 6) = v8;
  return *(_QWORD *)(a1 + 32) + 88 * v8 + 24;
}
