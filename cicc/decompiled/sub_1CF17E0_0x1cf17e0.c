// Function: sub_1CF17E0
// Address: 0x1cf17e0
//
__int64 *__fastcall sub_1CF17E0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 *v7; // rax
  __int64 *v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rdi
  unsigned __int64 v11; // r8
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rbx
  char v16; // cl
  __int64 v18; // rdx
  unsigned int v19; // edi
  __int64 *v20; // rcx
  __int64 v21; // [rsp+8h] [rbp-78h]
  unsigned __int64 v22; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+18h] [rbp-68h] BYREF
  __int64 v25; // [rsp+20h] [rbp-60h]
  __int64 v26; // [rsp+28h] [rbp-58h]
  _QWORD v27[3]; // [rsp+30h] [rbp-50h] BYREF
  char v28; // [rsp+48h] [rbp-38h]

  v5 = *a2;
  v24 = 0;
  v6 = *(_QWORD *)(v5 + 80);
  v25 = 0;
  v26 = 0;
  if ( v6 )
    v6 -= 24;
  v7 = *(__int64 **)(a3 + 8);
  if ( *(__int64 **)(a3 + 16) != v7 )
    goto LABEL_4;
  v18 = *(unsigned int *)(a3 + 28);
  v8 = &v7[v18];
  v19 = v18;
  if ( v7 == v8 )
  {
LABEL_28:
    if ( v19 < *(_DWORD *)(a3 + 24) )
    {
      *(_DWORD *)(a3 + 28) = v19 + 1;
      *v8 = v6;
      ++*(_QWORD *)a3;
LABEL_18:
      v8 = v27;
      v27[0] = v6;
      v28 = 0;
      sub_144A690(&v24, (__int64)v27);
      goto LABEL_5;
    }
LABEL_4:
    v8 = (__int64 *)v6;
    sub_16CCBA0(a3, v6);
    if ( !(_BYTE)v9 )
      goto LABEL_5;
    goto LABEL_18;
  }
  v20 = 0;
  while ( 1 )
  {
    v9 = *v7;
    if ( v6 == *v7 )
      break;
    if ( v9 == -2 )
      v20 = v7;
    if ( v8 == ++v7 )
    {
      if ( !v20 )
        goto LABEL_28;
      *v20 = v6;
      --*(_DWORD *)(a3 + 32);
      ++*(_QWORD *)a3;
      goto LABEL_18;
    }
  }
LABEL_5:
  v10 = v24;
  v11 = v25 - v24;
  if ( v25 == v24 )
  {
    v13 = 0;
LABEL_27:
    v15 = v13;
    goto LABEL_13;
  }
  if ( v11 > 0x7FFFFFFFFFFFFFE0LL )
    sub_4261EA(v24, v8, v9);
  v21 = v25 - v24;
  v12 = sub_22077B0(v25 - v24);
  v10 = v24;
  v13 = v12;
  v11 = v12 + v21;
  if ( v25 == v24 )
    goto LABEL_27;
  v14 = v24;
  v15 = v12 + v25 - v24;
  do
  {
    if ( v12 )
    {
      *(_QWORD *)v12 = *(_QWORD *)v14;
      v16 = *(_BYTE *)(v14 + 24);
      *(_BYTE *)(v12 + 24) = v16;
      if ( v16 )
        *(__m128i *)(v12 + 8) = _mm_loadu_si128((const __m128i *)(v14 + 8));
    }
    v12 += 32;
    v14 += 32;
  }
  while ( v15 != v12 );
LABEL_13:
  if ( v10 )
  {
    v22 = v11;
    j_j___libc_free_0(v10, v26 - v10);
    v11 = v22;
  }
  *a1 = a3;
  a1[1] = v13;
  a1[2] = v15;
  a1[4] = a3;
  a1[3] = v11;
  a1[5] = 0;
  a1[6] = 0;
  a1[7] = 0;
  return a1;
}
