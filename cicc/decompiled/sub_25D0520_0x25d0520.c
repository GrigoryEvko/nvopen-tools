// Function: sub_25D0520
// Address: 0x25d0520
//
unsigned __int64 __fastcall sub_25D0520(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  __int64 v6; // rax
  unsigned int v7; // esi
  __int64 v8; // rdi
  int v9; // eax
  int v10; // edx
  int v11; // ecx
  unsigned int v12; // r9d
  unsigned int v13; // eax
  int *v14; // r8
  int v15; // r10d
  unsigned __int64 result; // rax
  int v17; // r8d
  __int64 v18; // r13
  int *v19; // r8
  int v20; // edx
  int v21; // r11d
  int *v22; // r10
  int v23; // eax
  int v24; // edx
  int v25; // r11d
  int *v26; // [rsp+8h] [rbp-48h] BYREF
  __m128i v27; // [rsp+10h] [rbp-40h] BYREF
  __int64 v28; // [rsp+20h] [rbp-30h]

  v5 = *(_QWORD *)a1;
  v27.m128i_i64[0] = a2;
  v27.m128i_i64[1] = a3;
  v28 = a4;
  v26 = (int *)*(unsigned int *)(v5 + 40);
  v6 = sub_25D0030(v5, &v27, (__int64 *)&v26);
  v7 = *(_DWORD *)(a1 + 32);
  v8 = *(_QWORD *)(a1 + 16);
  v9 = *(_DWORD *)(v6 + 24);
  v10 = 2 * v9;
  v27.m128i_i32[0] = 2 * v9;
  v11 = (2 * v9) | 1;
  v27.m128i_i32[1] = v11;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 8);
    v18 = a1 + 8;
    v26 = 0;
    goto LABEL_16;
  }
  v12 = v7 - 1;
  v13 = (v7 - 1) & (74 * v9);
  v14 = (int *)(v8 + 4LL * v13);
  v15 = *v14;
  if ( v10 == *v14 )
  {
LABEL_3:
    result = v8 + 4LL * v7;
    if ( v14 != (int *)result )
      return result;
  }
  else
  {
    v17 = 1;
    while ( v15 != -1 )
    {
      v25 = v17 + 1;
      v13 = v12 & (v17 + v13);
      v14 = (int *)(v8 + 4LL * v13);
      v15 = *v14;
      if ( v10 == *v14 )
        goto LABEL_3;
      v17 = v25;
    }
  }
  v18 = a1 + 8;
  result = v12 & (37 * v11);
  v19 = (int *)(v8 + 4 * result);
  v20 = *v19;
  if ( v11 != *v19 )
  {
    v21 = 1;
    v22 = 0;
    while ( v20 != -1 )
    {
      if ( v22 || v20 != -2 )
        v19 = v22;
      result = v12 & (v21 + (_DWORD)result);
      v20 = *(_DWORD *)(v8 + 4LL * (unsigned int)result);
      if ( v11 == v20 )
        return result;
      ++v21;
      v22 = v19;
      v19 = (int *)(v8 + 4LL * (unsigned int)result);
    }
    v23 = *(_DWORD *)(a1 + 24);
    if ( !v22 )
      v22 = v19;
    ++*(_QWORD *)(a1 + 8);
    v24 = v23 + 1;
    v26 = v22;
    if ( 4 * (v23 + 1) < 3 * v7 )
    {
      result = v7 - *(_DWORD *)(a1 + 28) - v24;
      if ( (unsigned int)result > v7 >> 3 )
      {
LABEL_18:
        *(_DWORD *)(a1 + 24) = v24;
        if ( *v22 != -1 )
          --*(_DWORD *)(a1 + 28);
        *v22 = v11;
        return result;
      }
LABEL_17:
      sub_A08C50(v18, v7);
      sub_22B31A0(v18, &v27.m128i_i32[1], &v26);
      result = *(unsigned int *)(a1 + 24);
      v11 = v27.m128i_i32[1];
      v22 = v26;
      v24 = result + 1;
      goto LABEL_18;
    }
LABEL_16:
    v7 *= 2;
    goto LABEL_17;
  }
  return result;
}
