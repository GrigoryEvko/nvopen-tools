// Function: sub_1548EC0
// Address: 0x1548ec0
//
void __fastcall sub_1548EC0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __m128i a7)
{
  __int64 *v7; // r13
  __int64 v10; // rsi
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdi
  __int64 v17; // rsi
  int v18; // ecx
  __int64 v19; // rax
  __int64 v20; // rdx

  if ( (__int64 *)a1 != a2 )
  {
    v7 = (__int64 *)(a1 + 16);
    while ( a2 != v7 )
    {
      while ( 1 )
      {
        v10 = *v7;
        v11 = sub_1548C70(a7.m128i_i64, *v7, *(_QWORD *)a1);
        v16 = (__int64)v7;
        v7 += 2;
        if ( !v11 )
          break;
        v17 = *(v7 - 2);
        v18 = *((_DWORD *)v7 - 2);
        v19 = (v16 - a1) >> 4;
        if ( v16 - a1 > 0 )
        {
          do
          {
            v20 = *(_QWORD *)(v16 - 16);
            v16 -= 16;
            *(_QWORD *)(v16 + 16) = v20;
            *(_DWORD *)(v16 + 24) = *(_DWORD *)(v16 + 8);
            --v19;
          }
          while ( v19 );
        }
        *(_QWORD *)a1 = v17;
        *(_DWORD *)(a1 + 8) = v18;
        if ( a2 == v7 )
          return;
      }
      sub_1548E60(v16, v10, v12, v13, v14, v15, _mm_loadu_si128(&a7).m128i_i64[0]);
    }
  }
}
