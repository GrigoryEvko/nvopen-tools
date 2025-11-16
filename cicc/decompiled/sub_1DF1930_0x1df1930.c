// Function: sub_1DF1930
// Address: 0x1df1930
//
__int64 __fastcall sub_1DF1930(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned int v10; // ebx
  unsigned int v11; // r12d
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rax
  unsigned int v15; // r15d
  __int64 v16; // rbx
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // r13
  unsigned int v20; // eax
  __int64 *v21; // r12
  __int64 v22; // r13
  __int64 v23; // r15
  __int64 v24; // rsi
  unsigned int v26; // eax
  unsigned int v27; // [rsp+0h] [rbp-70h]
  unsigned int v28; // [rsp+4h] [rbp-6Ch]
  __m128i v30; // [rsp+10h] [rbp-60h] BYREF
  __int64 v31; // [rsp+20h] [rbp-50h]
  __int64 v32; // [rsp+28h] [rbp-48h]
  __m128i v33[4]; // [rsp+30h] [rbp-40h] BYREF

  v7 = *a3;
  v8 = *((unsigned int *)a3 + 2);
  v31 = a2;
  v9 = *(_QWORD *)(v7 + 8 * v8 - 8);
  v30.m128i_i64[0] = a5;
  v30.m128i_i64[1] = a6;
  v32 = v9;
  if ( v8 == 1 )
  {
    v10 = 0;
  }
  else
  {
    v10 = 0;
    v11 = 0;
    v12 = 0;
    while ( 1 )
    {
      v10 += sub_1F4BF20(a1 + 360, *(_QWORD *)(v7 + 8 * v12), 1);
      v12 = ++v11;
      if ( v11 >= (unsigned __int64)*((unsigned int *)a3 + 2) - 1 )
        break;
      v7 = *a3;
    }
  }
  v13 = *(_QWORD *)(v32 + 32);
  v14 = *(unsigned int *)(v32 + 40);
  v33[0] = _mm_load_si128(&v30);
  if ( v13 != v13 + 40 * v14 )
  {
    v28 = v10;
    v15 = 0;
    v16 = v13 + 40 * v14;
    v30.m128i_i64[0] = a1 + 360;
    do
    {
      if ( !*(_BYTE *)v13 )
      {
        v17 = *(_DWORD *)(v13 + 8);
        if ( v17 < 0 && (*(_BYTE *)(v13 + 3) & 0x10) != 0 )
        {
          v18 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 328) + 24LL) + 16LL * (v17 & 0x7FFFFFFF) + 8) + 32LL);
          v19 = *(_QWORD *)(v18 + 16);
          if ( v19 && (unsigned __int8)sub_1E81160(v33, v31, *(_QWORD *)(v18 + 16)) )
          {
            v27 = sub_1E165A0(v19, *(unsigned int *)(v13 + 8), 0, 0);
            v26 = sub_1E16810(v32, *(unsigned int *)(v13 + 8), 0, 0, 0);
            v20 = sub_1F4BB70(v30.m128i_i64[0], v32, v26, v19, v27);
          }
          else
          {
            v20 = sub_1F4BF20(v30.m128i_i64[0], v32, 1);
          }
          if ( v15 < v20 )
            v15 = v20;
        }
      }
      v13 += 40;
    }
    while ( v16 != v13 );
    v10 = v15 + v28;
  }
  v21 = *(__int64 **)a4;
  v22 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  if ( v22 == *(_QWORD *)a4 )
  {
    v23 = 0;
  }
  else
  {
    LODWORD(v23) = 0;
    do
    {
      v24 = *v21++;
      v23 = (unsigned int)sub_1F4BF20(a1 + 360, v24, 1) + (unsigned int)v23;
    }
    while ( (__int64 *)v22 != v21 );
  }
  return (v23 << 32) | v10;
}
