// Function: sub_1E69F60
// Address: 0x1e69f60
//
__int64 __fastcall sub_1E69F60(__int64 a1, __int64 a2)
{
  __int64 (*v2)(); // rax
  __int64 v3; // rax
  __int64 result; // rax
  __m128i v5; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v6; // [rsp+10h] [rbp-20h]

  v2 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 112LL);
  if ( v2 == sub_1D00B10 )
    BUG();
  v3 = v2();
  (*(void (__fastcall **)(__m128i *, __int64, __int64))(*(_QWORD *)v3 + 64LL))(&v5, v3, a2);
  _libc_free(*(_QWORD *)(a1 + 304));
  result = v6;
  *(__m128i *)(a1 + 304) = _mm_loadu_si128(&v5);
  *(_DWORD *)(a1 + 320) = result;
  return result;
}
