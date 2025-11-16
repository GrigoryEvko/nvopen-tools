// Function: sub_39EF5A0
// Address: 0x39ef5a0
//
__int64 __fastcall sub_39EF5A0(_DWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rdx
  __m128i *v16; // rax
  __int64 v17; // rax
  __int64 result; // rax
  int v19; // edx
  __int64 v20; // rsi
  __int64 v21; // r8
  const char *v22; // [rsp+0h] [rbp-60h] BYREF
  char v23; // [rsp+10h] [rbp-50h]
  char v24; // [rsp+11h] [rbp-4Fh]
  __m128i v25; // [rsp+20h] [rbp-40h] BYREF
  __m128i v26; // [rsp+30h] [rbp-30h] BYREF

  v6 = **((_QWORD **)a1 + 33);
  v26.m128i_i16[0] = 257;
  v24 = 1;
  v22 = ".comment";
  v23 = 3;
  v9 = sub_38C3B80(v6, (__int64)&v22, 1, 48, 1, (__int64)&v25, -1, 0);
  v10 = (unsigned int)a1[30];
  if ( (_DWORD)v10 )
  {
    v11 = (__int64 *)(*((_QWORD *)a1 + 14) + 32LL * (unsigned int)v10 - 32);
    v12 = v11[2];
    v13 = v11[3];
    v14 = *v11;
    v15 = v11[1];
  }
  else
  {
    v15 = 0;
    v14 = 0;
    v13 = 0;
    v12 = 0;
  }
  v25.m128i_i64[0] = v14;
  v25.m128i_i64[1] = v15;
  v26.m128i_i64[0] = v12;
  v26.m128i_i64[1] = v13;
  if ( a1[31] <= (unsigned int)v10 )
  {
    sub_16CD150((__int64)(a1 + 28), a1 + 32, 0, 32, v7, v8);
    v10 = (unsigned int)a1[30];
  }
  v16 = (__m128i *)(*((_QWORD *)a1 + 14) + 32 * v10);
  *v16 = _mm_loadu_si128(&v25);
  v16[1] = _mm_loadu_si128(&v26);
  v17 = *(_QWORD *)a1;
  ++a1[30];
  (*(void (__fastcall **)(_DWORD *, __int64, _QWORD))(v17 + 160))(a1, v9, 0);
  if ( !*((_BYTE *)a1 + 320) )
  {
    (*(void (__fastcall **)(_DWORD *, _QWORD, __int64))(*(_QWORD *)a1 + 424LL))(a1, 0, 1);
    *((_BYTE *)a1 + 320) = 1;
  }
  (*(void (__fastcall **)(_DWORD *, __int64, __int64))(*(_QWORD *)a1 + 400LL))(a1, a2, a3);
  (*(void (__fastcall **)(_DWORD *, _QWORD, __int64))(*(_QWORD *)a1 + 424LL))(a1, 0, 1);
  result = (unsigned int)a1[30];
  v19 = result;
  if ( (unsigned int)result > 1 )
  {
    result = *((_QWORD *)a1 + 14) + 32 * result;
    v20 = *(_QWORD *)(result - 64);
    v21 = *(_QWORD *)(result - 56);
    if ( *(_QWORD *)(result - 24) != v21 || *(_QWORD *)(result - 32) != v20 )
    {
      result = (*(__int64 (__fastcall **)(_DWORD *, __int64, __int64))(*(_QWORD *)a1 + 152LL))(a1, v20, v21);
      v19 = a1[30];
    }
    a1[30] = v19 - 1;
  }
  return result;
}
