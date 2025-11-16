// Function: sub_CB42E0
// Address: 0xcb42e0
//
__int64 __fastcall sub_CB42E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // ecx
  _QWORD *v5; // rsi
  __int64 **v6; // rax
  __int64 *v7; // rdx
  __int64 **v8; // rbx
  __int64 **v9; // r13
  __m128i *v10; // rcx
  __m128i *v11; // rsi
  __int64 v12; // rdx
  __int64 *v13; // rdx
  __int64 **v14; // rax
  __int64 *v15; // rdx
  __int64 *v17; // rsi
  __m128i v19; // [rsp+0h] [rbp-50h] BYREF
  char v20; // [rsp+20h] [rbp-30h]
  char v21; // [rsp+21h] [rbp-2Fh]

  v3 = *(_QWORD *)(a2 + 672);
  if ( *(_DWORD *)(*(_QWORD *)v3 + 32LL) == 4 )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    v4 = *(_DWORD *)(v3 + 16);
    if ( v4 )
    {
      v5 = *(_QWORD **)(v3 + 8);
      if ( *v5 != -8 && *v5 )
      {
        v8 = *(__int64 ***)(v3 + 8);
      }
      else
      {
        v6 = (__int64 **)(v5 + 1);
        do
        {
          do
          {
            v7 = *v6;
            v8 = v6++;
          }
          while ( v7 == (__int64 *)-8LL );
        }
        while ( !v7 );
      }
      v9 = (__int64 **)&v5[v4];
      if ( v9 != v8 )
      {
        v10 = 0;
        v11 = 0;
        while ( 1 )
        {
          v12 = **v8;
          v19.m128i_i64[0] = (__int64)(*v8 + 4);
          v19.m128i_i64[1] = v12;
          if ( v11 == v10 )
          {
            sub_C677B0((const __m128i **)a1, v11, &v19);
          }
          else
          {
            if ( v11 )
            {
              *v11 = _mm_loadu_si128(&v19);
              v11 = *(__m128i **)(a1 + 8);
            }
            *(_QWORD *)(a1 + 8) = v11 + 1;
          }
          v13 = v8[1];
          v14 = v8 + 1;
          if ( !v13 || v13 == (__int64 *)-8LL )
          {
            do
            {
              do
              {
                v15 = v14[1];
                ++v14;
              }
              while ( v15 == (__int64 *)-8LL );
            }
            while ( !v15 );
          }
          if ( v14 == v9 )
            break;
          v11 = *(__m128i **)(a1 + 8);
          v10 = *(__m128i **)(a1 + 16);
          v8 = v14;
        }
      }
    }
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    v17 = *(__int64 **)(a2 + 672);
    v21 = 1;
    v19.m128i_i64[0] = (__int64)"not a mapping";
    v20 = 3;
    sub_CB1040(a2, v17, (__int64)&v19);
  }
  return a1;
}
