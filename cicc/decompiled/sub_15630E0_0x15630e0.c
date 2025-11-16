// Function: sub_15630E0
// Address: 0x15630e0
//
__int64 __fastcall sub_15630E0(__int64 *a1, __int64 *a2, _BYTE *a3, size_t a4)
{
  __int64 v6; // r12
  __m128i v8; // [rsp+0h] [rbp-90h] BYREF
  _QWORD *v9; // [rsp+18h] [rbp-78h]

  if ( !sub_155F280(a1, a3, a4) )
    return *a1;
  sub_1563030(&v8, *a1);
  sub_1560780((__int64)&v8, a3, a4);
  v6 = sub_1560BF0(a2, &v8);
  sub_155CC10(v9);
  return v6;
}
