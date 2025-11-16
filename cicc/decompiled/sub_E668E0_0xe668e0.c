// Function: sub_E668E0
// Address: 0xe668e0
//
__int64 __fastcall sub_E668E0(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned __int8 *v3; // rax
  __int64 result; // rax
  _QWORD *v5; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v6[2]; // [rsp+10h] [rbp-30h] BYREF
  __int64 (__fastcall *v7)(const __m128i **, const __m128i *, int); // [rsp+20h] [rbp-20h]
  __int64 (__fastcall *v8)(); // [rsp+28h] [rbp-18h]

  v3 = *(unsigned __int8 **)(a1 + 2368);
  v5 = a2;
  if ( v3 )
  {
    result = *v3;
    if ( (result & 8) != 0 )
      return result;
    if ( (result & 4) != 0 )
      return sub_E66880(a1, a2, a3);
  }
  v6[1] = a3;
  v6[0] = &v5;
  v8 = sub_E6F790;
  v7 = sub_E62B80;
  sub_E664E0(a1, v5, (__int64)v6);
  result = (__int64)v7;
  if ( v7 )
    return v7((const __m128i **)v6, (const __m128i *)v6, 3);
  return result;
}
