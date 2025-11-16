// Function: sub_299F9B0
// Address: 0x299f9b0
//
unsigned __int64 *__fastcall sub_299F9B0(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4)
{
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 i; // r15
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // r10
  __int64 v15; // rsi
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rax
  __int64 v18; // rdi
  __int64 j; // r9
  bool v20; // cf
  __int64 v21; // rax
  __int64 v22; // r15
  unsigned __int64 v23; // r11
  unsigned __int64 v24; // r8
  unsigned __int64 v25; // rax
  bool v26; // cf
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  unsigned __int64 *result; // rax
  unsigned __int64 v30; // r14
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rdx

  v8 = *(unsigned int *)(a2 + 8);
  if ( !v8 )
  {
    sub_299F8F0(a2, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_299DCE0);
    v31 = *(_QWORD *)a2;
    *a1 = a3;
    v32 = *(_QWORD *)(v31 + 24);
    v33 = v32;
    if ( a3 >= v32 )
      v33 = a3;
    if ( a4 >= v32 )
      v32 = a4;
    a1[1] = v33;
    if ( v32 >= a3 )
      a3 = v32;
    v16 = a3;
    goto LABEL_39;
  }
  v9 = 0;
  for ( i = 0; i != v8; ++i )
  {
    v11 = 16;
    v12 = v9 + *(_QWORD *)a2;
    if ( *(_QWORD *)(v12 + 24) >= 0x10u )
      v11 = *(_QWORD *)(v12 + 24);
    v9 += 56;
    *(_QWORD *)(v12 + 24) = v11;
  }
  sub_299F8F0(a2, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_299DCE0);
  v13 = *(_QWORD *)a2;
  *a1 = a3;
  v14 = 2 * a3;
  v15 = i - 1;
  v16 = *(_QWORD *)(v13 + 24);
  v17 = v16;
  if ( a3 >= v16 )
    v17 = a3;
  if ( a4 >= v16 )
    v16 = a4;
  a1[1] = v17;
  if ( v16 < a3 )
    v16 = a3;
  v18 = 0;
  for ( j = 0; ; ++j )
  {
    v22 = v13 + v18;
    v23 = *(_QWORD *)(v13 + v18 + 8);
    if ( j == v15 )
      break;
    v24 = a3;
    if ( *(_QWORD *)(v13 + v18 + 80) >= a3 )
      v24 = *(_QWORD *)(v13 + v18 + 80);
    if ( v23 <= 4 )
    {
      *(_QWORD *)(v22 + 40) = v16;
      v20 = v14 < 0x10;
      v21 = 16;
LABEL_14:
      if ( !v20 )
        v21 = 2 * a3;
      v18 += 56;
      v16 += v24 * ((v21 - 1) / v24 + 1);
      goto LABEL_17;
    }
    if ( v23 <= 0x10 )
    {
      *(_QWORD *)(v22 + 40) = v16;
      v20 = v14 < 0x20;
      v21 = 32;
      goto LABEL_14;
    }
LABEL_26:
    v25 = v23 + 32;
    if ( v23 > 0x80 )
    {
      v25 = v23 + 64;
      if ( v23 > 0x200 )
      {
        v25 = v23 + 256;
        if ( v23 <= 0x1000 )
          v25 = v23 + 128;
      }
    }
    *(_QWORD *)(v22 + 40) = v16;
    if ( v25 < v14 )
      v25 = 2 * a3;
    v18 += 56;
    v16 += v24 * ((v25 != 0) + (v25 - (v25 != 0)) / v24);
    if ( j == v15 )
      goto LABEL_39;
LABEL_17:
    v13 = *(_QWORD *)a2;
  }
  if ( v23 <= 4 )
  {
    *(_QWORD *)(v22 + 40) = v16;
    v26 = v14 < 0x10;
    v27 = 16;
    goto LABEL_36;
  }
  if ( v23 > 0x10 )
  {
    v24 = a3;
    goto LABEL_26;
  }
  *(_QWORD *)(v22 + 40) = v16;
  v26 = v14 < 0x20;
  v27 = 32;
LABEL_36:
  if ( !v26 )
    v27 = 2 * a3;
  v16 += ((v27 - 1) / a3 + 1) * a3;
LABEL_39:
  v28 = v16 % a4;
  result = a1;
  v30 = v16 + a4 - v16 % a4;
  if ( v28 )
    v16 = v30;
  a1[2] = v16;
  return result;
}
