// Function: sub_37236D0
// Address: 0x37236d0
//
unsigned __int64 __fastcall sub_37236D0(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // rbx
  __m128i v4; // [rsp+10h] [rbp-30h] BYREF
  __int128 v5; // [rsp+20h] [rbp-20h] BYREF

  v1 = sub_3214EE0(a1);
  if ( v1 )
  {
    v2 = v1;
    sub_3215160((__int64)&v5, v1, 60);
    if ( !(_DWORD)v5 )
      return *(unsigned int *)(v2 + 16);
  }
  v4 = 0u;
  return _mm_loadu_si128(&v4).m128i_u64[0];
}
