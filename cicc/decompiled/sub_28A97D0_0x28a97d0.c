// Function: sub_28A97D0
// Address: 0x28a97d0
//
char __fastcall sub_28A97D0(
        _QWORD *a1,
        _QWORD **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __m128i a9)
{
  char result; // al
  _QWORD *v14; // rax
  __int64 v15; // r8
  _QWORD *v16; // rdi
  _QWORD *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // rbx
  __int64 v21; // r14
  unsigned __int8 *v22; // rsi
  _QWORD *v23; // rdi
  __m128i v24; // xmm0
  __m128i v25; // xmm1
  __m128i v26; // xmm2
  __m128i v27; // [rsp+8h] [rbp-98h] BYREF
  __m128i v28; // [rsp+18h] [rbp-88h] BYREF
  __m128i v29; // [rsp+28h] [rbp-78h] BYREF
  __m128i v30[3]; // [rsp+40h] [rbp-60h] BYREF
  char v31; // [rsp+70h] [rbp-30h]

  if ( *(_BYTE *)a4 == 26 )
  {
    result = 1;
    if ( *(_QWORD *)(a4 + 64) == *(_QWORD *)(a3 + 64) )
    {
      v19 = *(_QWORD *)(a3 + 40);
      v20 = a4 + 32;
      v27 = a7;
      v28 = a8;
      v29 = a9;
      if ( a4 + 32 != v19 )
      {
        v21 = (__int64)(a2 + 1);
        do
        {
          if ( !v19 )
            BUG();
          if ( *(_BYTE *)(v19 - 32) != 26 )
          {
            v22 = *(unsigned __int8 **)(v19 + 40);
            v23 = *a2;
            v24 = _mm_loadu_si128(&v27);
            v25 = _mm_loadu_si128(&v28);
            v31 = 1;
            v26 = _mm_loadu_si128(&v29);
            v30[0] = v24;
            v30[1] = v25;
            v30[2] = v26;
            if ( (sub_CF63E0(v23, v22, v30, v21) & 2) != 0 )
              break;
          }
          v19 = *(_QWORD *)(v19 + 8);
        }
        while ( v20 != v19 );
      }
      return v19 != v20;
    }
  }
  else
  {
    v14 = sub_103E0E0(a1);
    v15 = *v14;
    v16 = v14;
    v17 = (_QWORD *)(a4 - 64);
    if ( *(_BYTE *)a4 == 26 )
      v17 = (_QWORD *)(a4 - 32);
    v18 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, __m128i *, _QWORD **))(v15 + 24))(v16, *v17, &a7, a2);
    return sub_1041420((__int64)a1, v18, a3) ^ 1;
  }
  return result;
}
