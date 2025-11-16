// Function: sub_21589E0
// Address: 0x21589e0
//
__int64 __fastcall sub_21589E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v6; // r12d
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // rax
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rax
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // r12
  __int64 v20; // rax
  const void *v21; // [rsp+0h] [rbp-70h]
  int v22; // [rsp+Ch] [rbp-64h]
  __int64 v23; // [rsp+10h] [rbp-60h]
  __int64 v24; // [rsp+18h] [rbp-58h]
  __m128i v25; // [rsp+20h] [rbp-50h] BYREF
  __int16 v26; // [rsp+30h] [rbp-40h]

  *(_DWORD *)a3 = **(unsigned __int16 **)(a2 + 16);
  if ( **(_WORD **)(a2 + 16) == 189 )
  {
    v14 = *(_QWORD *)(a1 + 248);
    v15 = *(_BYTE **)(*(_QWORD *)(a2 + 32) + 24LL);
    v26 = 257;
    if ( *v15 )
    {
      v25.m128i_i64[0] = (__int64)v15;
      LOBYTE(v26) = 3;
    }
    v16 = sub_38BF510(v14, &v25);
    v19 = sub_38CF310(v16, 0, *(_QWORD *)(a1 + 248), 0);
    v20 = *(unsigned int *)(a3 + 24);
    if ( (unsigned int)v20 >= *(_DWORD *)(a3 + 28) )
    {
      sub_16CD150(a3 + 16, (const void *)(a3 + 32), 0, 16, v17, v18);
      v20 = *(unsigned int *)(a3 + 24);
    }
    result = *(_QWORD *)(a3 + 16) + 16 * v20;
    *(_QWORD *)result = 4;
    *(_QWORD *)(result + 8) = v19;
    ++*(_DWORD *)(a3 + 24);
  }
  else
  {
    result = *(unsigned int *)(a2 + 40);
    v6 = 0;
    v24 = 0;
    v22 = result;
    v21 = (const void *)(a3 + 32);
    if ( (_DWORD)result )
    {
      do
      {
        while ( 1 )
        {
          v9 = *(_QWORD *)(a2 + 32);
          v25 = 0u;
          v23 = v9;
          if ( !sub_214D110(a1, a2, v6, (__int64)&v25) )
            break;
          v12 = *(unsigned int *)(a3 + 24);
          if ( (unsigned int)v12 >= *(_DWORD *)(a3 + 28) )
          {
            sub_16CD150(a3 + 16, v21, 0, 16, v10, v11);
            v12 = *(unsigned int *)(a3 + 24);
          }
          result = *(_QWORD *)(a3 + 16) + 16 * v12;
          ++v6;
          v24 += 40;
          *(__m128i *)result = _mm_load_si128(&v25);
          ++*(_DWORD *)(a3 + 24);
          if ( v6 == v22 )
            return result;
        }
        result = sub_2158820(a1, v24 + v23, (__int64)&v25);
        if ( (_BYTE)result )
        {
          v13 = *(unsigned int *)(a3 + 24);
          if ( (unsigned int)v13 >= *(_DWORD *)(a3 + 28) )
          {
            sub_16CD150(a3 + 16, v21, 0, 16, v7, v8);
            v13 = *(unsigned int *)(a3 + 24);
          }
          result = *(_QWORD *)(a3 + 16) + 16 * v13;
          *(__m128i *)result = _mm_load_si128(&v25);
          ++*(_DWORD *)(a3 + 24);
        }
        v24 += 40;
        ++v6;
      }
      while ( v6 != v22 );
    }
  }
  return result;
}
