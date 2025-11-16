// Function: sub_A3EC30
// Address: 0xa3ec30
//
void __fastcall sub_A3EC30(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        _BYTE *a8)
{
  __int64 *v8; // r13
  __int64 v11; // rsi
  char v12; // al
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // rdi
  __int64 v18; // rsi
  int v19; // ecx
  __int64 v20; // rax
  __int64 v21; // rdx
  __m128i v22; // [rsp-68h] [rbp-68h]

  if ( (__int64 *)a1 != a2 )
  {
    v8 = (__int64 *)(a1 + 16);
    while ( a2 != v8 )
    {
      while ( 1 )
      {
        v11 = *v8;
        v12 = sub_A3D900((__int64)&a7, *v8, *(_QWORD *)a1);
        v17 = v8;
        v8 += 2;
        if ( !v12 )
          break;
        v18 = *(v8 - 2);
        v19 = *((_DWORD *)v8 - 2);
        v20 = ((__int64)v17 - a1) >> 4;
        if ( (__int64)v17 - a1 > 0 )
        {
          do
          {
            v21 = *(v17 - 2);
            v17 -= 2;
            v17[2] = v21;
            *((_DWORD *)v17 + 6) = *((_DWORD *)v17 + 2);
            --v20;
          }
          while ( v20 );
        }
        *(_QWORD *)a1 = v18;
        *(_DWORD *)(a1 + 8) = v19;
        if ( a2 == v8 )
          return;
      }
      v22 = _mm_loadu_si128(&a7);
      sub_A3EA00(v17, v11, v13, v14, v15, v16, v22.m128i_i64[0], (unsigned int *)v22.m128i_i64[1], a8);
    }
  }
}
