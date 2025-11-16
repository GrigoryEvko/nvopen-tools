// Function: sub_38AA300
// Address: 0x38aa300
//
__int64 __fastcall sub_38AA300(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  int v9; // eax
  double v10; // xmm4_8
  double v11; // xmm5_8
  const __m128i *v12; // rsi
  bool v13; // zf
  unsigned int v14; // r14d
  unsigned __int64 v15; // rdi
  __m128i *v16; // r12
  const __m128i *v17; // rbx
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int32 v21; // [rsp+4h] [rbp-6Ch] BYREF
  __int64 v22; // [rsp+8h] [rbp-68h] BYREF
  __m128i v23; // [rsp+10h] [rbp-60h] BYREF
  const __m128i *v24; // [rsp+20h] [rbp-50h] BYREF
  __m128i *v25; // [rsp+28h] [rbp-48h]
  const __m128i *v26; // [rsp+30h] [rbp-40h]

  v9 = sub_3887100(a1 + 8);
  v24 = 0;
  *(_DWORD *)(a1 + 64) = v9;
  v25 = 0;
  v26 = 0;
  if ( v9 == 376 )
  {
    while ( 1 )
    {
      v14 = sub_38AA270(a1, &v21, &v22, a2, a3, a4, a5, v10, v11, a8, a9);
      if ( (_BYTE)v14 )
        break;
      v12 = v25;
      v23.m128i_i32[0] = v21;
      v23.m128i_i64[1] = v22;
      if ( v25 == v26 )
      {
        sub_38938B0((unsigned __int64 *)&v24, v25, &v23);
        if ( *(_DWORD *)(a1 + 64) != 376 )
          goto LABEL_9;
      }
      else
      {
        if ( v25 )
        {
          a2 = (__m128)_mm_loadu_si128(&v23);
          *v25 = (__m128i)a2;
          v12 = v25;
        }
        v13 = *(_DWORD *)(a1 + 64) == 376;
        v25 = (__m128i *)&v12[1];
        if ( !v13 )
          goto LABEL_9;
      }
    }
  }
  else
  {
LABEL_9:
    v14 = sub_389FA00(a1, v23.m128i_i64, 0);
    if ( !(_BYTE)v14 )
    {
      v15 = (unsigned __int64)v24;
      v16 = v25;
      if ( v25 == v24 )
        goto LABEL_14;
      v17 = v24;
      do
      {
        v18 = v17->m128i_i64[1];
        v19 = v17->m128i_i32[0];
        ++v17;
        sub_16267C0(v23.m128i_i64[0], v19, v18);
      }
      while ( v16 != v17 );
    }
  }
  v15 = (unsigned __int64)v24;
LABEL_14:
  if ( v15 )
    j_j___libc_free_0(v15);
  return v14;
}
