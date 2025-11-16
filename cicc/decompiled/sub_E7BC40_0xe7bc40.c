// Function: sub_E7BC40
// Address: 0xe7bc40
//
void __fastcall sub_E7BC40(_QWORD *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdi
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // rdx
  unsigned int v13; // edi
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  __m128i *v20; // rsi
  unsigned int v21; // [rsp+4h] [rbp-5Ch] BYREF
  unsigned int *v22; // [rsp+8h] [rbp-58h] BYREF
  __m128i v23; // [rsp+10h] [rbp-50h] BYREF
  __m128i v24; // [rsp+20h] [rbp-40h] BYREF
  __m128i v25; // [rsp+30h] [rbp-30h] BYREF

  v6 = a1[1];
  if ( *(_BYTE *)(v6 + 1792) )
  {
    v8 = sub_E6C430(v6, (__int64)a2, a3, a4, a5);
    (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1 + 208LL))(a1, v8, 0);
    v9 = a1[1];
    v24 = (__m128i)(unsigned __int64)v8;
    *(_BYTE *)(v9 + 1792) = 0;
    v10 = a1[1];
    v11 = *(_QWORD *)(v9 + 1776);
    v12 = *(_QWORD *)(v9 + 1784);
    v25.m128i_i64[0] = 0;
    v13 = *(_DWORD *)(v10 + 1912);
    v14 = *(_QWORD *)(v10 + 1744);
    v15 = v10 + 1736;
    v25.m128i_i8[8] = 0;
    v23.m128i_i64[0] = v11;
    v16 = v10 + 1736;
    v23.m128i_i64[1] = v12;
    v21 = v13;
    if ( !v14 )
      goto LABEL_9;
    do
    {
      while ( 1 )
      {
        v17 = *(_QWORD *)(v14 + 16);
        v18 = *(_QWORD *)(v14 + 24);
        if ( v13 <= *(_DWORD *)(v14 + 32) )
          break;
        v14 = *(_QWORD *)(v14 + 24);
        if ( !v18 )
          goto LABEL_7;
      }
      v16 = v14;
      v14 = *(_QWORD *)(v14 + 16);
    }
    while ( v17 );
LABEL_7:
    if ( v15 == v16 || v13 < *(_DWORD *)(v16 + 32) )
    {
LABEL_9:
      v22 = &v21;
      v16 = sub_E7A040((_QWORD *)(v10 + 1728), v16, &v22);
    }
    v22 = a2;
    v19 = sub_E7B8C0(v16 + 560, (__int64 *)&v22, v18, v17, v10, v15);
    v20 = *(__m128i **)(v19 + 8);
    if ( v20 == *(__m128i **)(v19 + 16) )
    {
      sub_E782B0((const __m128i **)v19, v20, &v23);
    }
    else
    {
      if ( v20 )
      {
        *v20 = _mm_loadu_si128(&v23);
        v20[1] = _mm_loadu_si128(&v24);
        v20[2] = _mm_loadu_si128(&v25);
        v20 = *(__m128i **)(v19 + 8);
      }
      *(_QWORD *)(v19 + 8) = v20 + 3;
    }
  }
}
