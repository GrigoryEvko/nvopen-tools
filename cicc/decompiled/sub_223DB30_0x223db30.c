// Function: sub_223DB30
// Address: 0x223db30
//
__int64 __fastcall sub_223DB30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, unsigned __int8 a6)
{
  __int64 v8; // rbx
  int v9; // ecx
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r11
  int v16; // r9d
  __int64 v17; // r15
  __int64 v18; // r14
  size_t v19; // r15
  void *v20; // rsp
  void *v21; // rax
  char v22; // cl
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v26; // [rsp+8h] [rbp-A8h]
  __m128i v27; // [rsp+40h] [rbp-70h]
  __int64 v28; // [rsp+50h] [rbp-60h]
  __int64 v29; // [rsp+58h] [rbp-58h]
  char v30; // [rsp+6Fh] [rbp-41h] BYREF
  __m128i v31; // [rsp+70h] [rbp-40h] BYREF

  v8 = a3;
  v9 = *(_DWORD *)(a4 + 24);
  if ( (v9 & 1) == 0 )
  {
    result = sub_2232B80(a1, a2, a3, a4, (char)a5, a6);
    v28 = result;
    v29 = v11;
    return result;
  }
  HIDWORD(v25) = a5;
  LODWORD(v26) = v9;
  v12 = sub_2232A70((__int64)&v30, (__int64 *)(a4 + 208));
  v13 = (unsigned int)v26;
  v14 = HIDWORD(v25);
  if ( a6 )
  {
    v15 = *(_QWORD *)(v12 + 40);
    v16 = *(_DWORD *)(v12 + 48);
  }
  else
  {
    v15 = *(_QWORD *)(v12 + 56);
    v16 = *(_DWORD *)(v12 + 64);
  }
  v17 = *(_QWORD *)(a4 + 16);
  v18 = v16;
  if ( v17 <= v16 )
  {
    *(_QWORD *)(a4 + 16) = 0;
    if ( (_BYTE)v8 )
      return a2;
LABEL_7:
    (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a2 + 96LL))(
      a2,
      v15,
      v18,
      v13,
      v14);
    return a2;
  }
  v19 = v17 - v16;
  HIDWORD(v25) = v26;
  v26 = v15;
  v20 = alloca(v19 + 8);
  v21 = memset(&v25, (char)v14, v19);
  v22 = BYTE4(v25);
  *(_QWORD *)(a4 + 16) = 0;
  if ( (v22 & 0xB0) != 0x20 )
  {
    if ( (_BYTE)v8 )
      return a2;
    v23 = (*(__int64 (__fastcall **)(__int64, void *, _QWORD))(*(_QWORD *)a2 + 96LL))(a2, v21, (int)v19);
    v15 = v26;
    if ( (int)v19 != v23 )
      return a2;
    goto LABEL_7;
  }
  v31.m128i_i64[0] = a2;
  v24 = v26;
  v26 = (__int64)v21;
  v31.m128i_i64[1] = v8;
  LOBYTE(v8) = 0;
  sub_223DAF0(&v31, v24, v18);
  v27 = _mm_loadu_si128(&v31);
  v31.m128i_i64[1] = v27.m128i_u8[8] | (unsigned __int64)v8;
  sub_223DAF0(&v31, v26, (int)v19);
  return v31.m128i_i64[0];
}
