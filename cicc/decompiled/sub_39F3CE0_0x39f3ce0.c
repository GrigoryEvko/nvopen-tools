// Function: sub_39F3CE0
// Address: 0x39f3ce0
//
__int64 __fastcall sub_39F3CE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, unsigned __int64 a6)
{
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rsi
  __m128i *v18; // rax
  __int64 result; // rax
  int v20; // edx
  __int64 v21; // rsi
  __int64 *v22; // rdx
  __int64 v23; // rdi
  __m128i v25; // [rsp+10h] [rbp-50h] BYREF
  __m128i v26; // [rsp+20h] [rbp-40h] BYREF

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2) )
  {
    v13 = *(unsigned int *)(a1 + 120);
    if ( (_DWORD)v13 )
    {
      v22 = (__int64 *)(*(_QWORD *)(a1 + 112) + 32LL * (unsigned int)v13 - 32);
      v17 = v22[2];
      v16 = v22[3];
      v15 = *v22;
      v14 = v22[1];
    }
    else
    {
      v14 = 0;
      v15 = 0;
      v16 = 0;
      v17 = 0;
    }
    v25.m128i_i64[0] = v15;
    v25.m128i_i64[1] = v14;
    v26.m128i_i64[0] = v17;
    v26.m128i_i64[1] = v16;
    if ( *(_DWORD *)(a1 + 124) <= (unsigned int)v13 )
    {
      sub_16CD150(a1 + 112, (const void *)(a1 + 128), 0, 32, v11, v12);
      v13 = *(unsigned int *)(a1 + 120);
    }
    v18 = (__m128i *)(*(_QWORD *)(a1 + 112) + 32 * v13);
    *v18 = _mm_loadu_si128(&v25);
    v18[1] = _mm_loadu_si128(&v26);
    ++*(_DWORD *)(a1 + 120);
    sub_38DC390(a1, a2, 0);
    if ( a3 )
    {
      sub_38D4400(a1, a5, 0, 1, 0);
      sub_39F3BB0(a1, a3, 0);
      sub_38DD110((__int64 *)a1, a4);
    }
    result = *(unsigned int *)(a1 + 120);
    v20 = result;
    if ( (unsigned int)result > 1 )
    {
      result = *(_QWORD *)(a1 + 112) + 32 * result;
      v21 = *(_QWORD *)(result - 64);
      if ( *(_QWORD *)(result - 24) != *(_QWORD *)(result - 56) || *(_QWORD *)(result - 32) != v21 )
      {
        result = sub_39F2FB0(a1, v21, *(_QWORD *)(result - 56));
        v20 = *(_DWORD *)(a1 + 120);
      }
      *(_DWORD *)(a1 + 120) = v20 - 1;
    }
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 8);
    v26.m128i_i16[0] = 259;
    v25.m128i_i64[0] = (__int64)"The usage of .zerofill is restricted to sections of ZEROFILL type. Use .zero or .space instead.";
    return (__int64)sub_38BE3D0(v23, a6, (__int64)&v25);
  }
  return result;
}
