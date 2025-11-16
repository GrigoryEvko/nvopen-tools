// Function: sub_36CDE10
// Address: 0x36cde10
//
__int64 *sub_36CDE10()
{
  __m128i *v0; // rax
  __int64 *v1; // r12
  __m128i v3; // [rsp+0h] [rbp-30h] BYREF
  __int64 (__fastcall *v4)(__m128i *, __int64, int); // [rsp+10h] [rbp-20h]
  __int64 (__fastcall *v5)(); // [rsp+18h] [rbp-18h]

  v0 = (__m128i *)sub_22077B0(0xD8u);
  v1 = (__int64 *)v0;
  if ( v0 )
  {
    v5 = sub_36CFEC0;
    v4 = (__int64 (__fastcall *)(__m128i *, __int64, int))sub_36CDAA0;
    sub_CF6B40(v0, &v3, 1);
    if ( v4 )
      v4(&v3, (__int64)&v3, 3);
    *v1 = (__int64)&unk_4A3B110;
  }
  return v1;
}
