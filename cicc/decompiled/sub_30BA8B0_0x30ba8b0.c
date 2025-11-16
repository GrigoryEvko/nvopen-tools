// Function: sub_30BA8B0
// Address: 0x30ba8b0
//
__int64 *__fastcall sub_30BA8B0(__int64 *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  bool v8; // zf
  __int64 v9; // rbx
  _QWORD *v10; // rax
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rcx
  __m128i *v13; // rax
  __m128i *v14; // r15
  const __m128i *v15; // rdx
  __m128i *v16; // rbx
  signed __int64 v18; // [rsp+8h] [rbp-78h]
  unsigned __int64 v19; // [rsp+8h] [rbp-78h]
  __int64 v20; // [rsp+10h] [rbp-70h]
  const __m128i *v21; // [rsp+18h] [rbp-68h] BYREF
  const __m128i *v22; // [rsp+20h] [rbp-60h]
  __int64 v23; // [rsp+28h] [rbp-58h]
  __m128i v24; // [rsp+30h] [rbp-50h] BYREF
  char v25; // [rsp+48h] [rbp-38h]

  v6 = a3;
  v8 = *(_BYTE *)(a3 + 28) == 0;
  v20 = a3;
  v21 = 0;
  v9 = a2->m128i_i64[0];
  v22 = 0;
  v23 = 0;
  if ( v8 )
    goto LABEL_17;
  v10 = *(_QWORD **)(a3 + 8);
  a4 = *(unsigned int *)(a3 + 20);
  a3 = (__int64)&v10[a4];
  if ( v10 == (_QWORD *)a3 )
  {
LABEL_16:
    if ( (unsigned int)a4 < *(_DWORD *)(v6 + 16) )
    {
      *(_DWORD *)(v6 + 20) = a4 + 1;
      *(_QWORD *)a3 = v9;
      ++*(_QWORD *)v6;
LABEL_18:
      a2 = &v24;
      v24.m128i_i64[0] = v9;
      v25 = 0;
      sub_30BA870((unsigned __int64 *)&v21, &v24);
      goto LABEL_6;
    }
LABEL_17:
    a2 = (__m128i *)v9;
    sub_C8CC70(v6, v9, a3, a4, a5, a6);
    if ( !(_BYTE)a3 )
      goto LABEL_6;
    goto LABEL_18;
  }
  while ( v9 != *v10 )
  {
    if ( (_QWORD *)a3 == ++v10 )
      goto LABEL_16;
  }
LABEL_6:
  v11 = (unsigned __int64)v21;
  v12 = (char *)v22 - (char *)v21;
  if ( v22 == v21 )
  {
    v14 = 0;
LABEL_20:
    v16 = v14;
    goto LABEL_13;
  }
  if ( v12 > 0x7FFFFFFFFFFFFFE0LL )
    sub_4261EA(v21, a2, a3);
  v18 = (char *)v22 - (char *)v21;
  v13 = (__m128i *)sub_22077B0((char *)v22 - (char *)v21);
  v11 = (unsigned __int64)v21;
  v14 = v13;
  v12 = (unsigned __int64)v13->m128i_u64 + v18;
  if ( v22 == v21 )
    goto LABEL_20;
  v15 = v21;
  v16 = (__m128i *)((char *)v13 + (char *)v22 - (char *)v21);
  do
  {
    if ( v13 )
    {
      *v13 = _mm_loadu_si128(v15);
      v13[1] = _mm_loadu_si128(v15 + 1);
    }
    v13 += 2;
    v15 += 2;
  }
  while ( v13 != v16 );
LABEL_13:
  if ( v11 )
  {
    v19 = v12;
    j_j___libc_free_0(v11);
    v12 = v19;
  }
  *a1 = v20;
  a1[1] = (__int64)v14;
  a1[2] = (__int64)v16;
  a1[4] = v6;
  a1[3] = v12;
  a1[5] = 0;
  a1[6] = 0;
  a1[7] = 0;
  return a1;
}
