// Function: sub_153CD00
// Address: 0x153cd00
//
void __fastcall sub_153CD00(__int64 a1, __int64 **a2, __int64 a3)
{
  __int64 **v3; // rbx
  bool v4; // r8
  __int64 **v5; // rax
  __int64 v6; // rsi
  int v7; // edi
  __int64 v8; // rdx
  __int64 *v9; // rcx
  __m128i v10; // xmm0
  __int64 **v11; // r15
  __int64 *v12; // rax
  __int64 v15; // [rsp+18h] [rbp-58h] BYREF
  __int64 v16; // [rsp+28h] [rbp-48h] BYREF
  __m128i v17; // [rsp+30h] [rbp-40h] BYREF

  v15 = a3;
  if ( (__int64 **)a1 != a2 )
  {
    v3 = (__int64 **)(a1 + 16);
    if ( (__int64 **)(a1 + 16) != a2 )
    {
      do
      {
        v4 = sub_153CA80((__int64)&v15, v3, (__int64 **)a1);
        v5 = v3;
        v3 += 2;
        if ( v4 )
        {
          v6 = (__int64)*(v3 - 2);
          v7 = *((_DWORD *)v3 - 2);
          v8 = ((__int64)v5 - a1) >> 4;
          if ( (__int64)v5 - a1 > 0 )
          {
            do
            {
              v9 = *(v5 - 2);
              v5 -= 2;
              v5[2] = v9;
              *((_DWORD *)v5 + 6) = *((_DWORD *)v5 + 2);
              --v8;
            }
            while ( v8 );
          }
          *(_QWORD *)a1 = v6;
          *(_DWORD *)(a1 + 8) = v7;
        }
        else
        {
          v10 = _mm_loadu_si128((const __m128i *)v3 - 1);
          v11 = v3 - 4;
          v16 = v15;
          v17 = v10;
          while ( sub_153CA80((__int64)&v16, (__int64 **)&v17, v11) )
          {
            v12 = *v11;
            v11 -= 2;
            v11[4] = v12;
            *((_DWORD *)v11 + 10) = *((_DWORD *)v11 + 6);
          }
          v11[2] = (__int64 *)v17.m128i_i64[0];
          *((_DWORD *)v11 + 6) = v17.m128i_i32[2];
        }
      }
      while ( a2 != v3 );
    }
  }
}
