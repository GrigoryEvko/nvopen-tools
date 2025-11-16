// Function: sub_E66880
// Address: 0xe66880
//
__int64 __fastcall sub_E66880(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 result; // rax
  _QWORD *v4; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v5[2]; // [rsp+10h] [rbp-30h] BYREF
  __int64 (__fastcall *v6)(const __m128i **, const __m128i *, int); // [rsp+20h] [rbp-20h]
  __int64 (__fastcall *v7)(); // [rsp+28h] [rbp-18h]

  *(_BYTE *)(a1 + 2376) = 1;
  v5[0] = &v4;
  v5[1] = a3;
  v7 = sub_E6FC60;
  v6 = sub_E62B50;
  v4 = a2;
  sub_E664E0(a1, a2, (__int64)v5);
  result = (__int64)v6;
  if ( v6 )
    return v6((const __m128i **)v5, (const __m128i *)v5, 3);
  return result;
}
