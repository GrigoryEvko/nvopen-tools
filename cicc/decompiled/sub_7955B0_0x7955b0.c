// Function: sub_7955B0
// Address: 0x7955b0
//
__int64 __fastcall sub_7955B0(__int64 a1, __m128i *a2)
{
  __int64 v2; // r12
  __int64 v4; // [rsp+0h] [rbp-F0h] BYREF
  unsigned __int64 v5; // [rsp+8h] [rbp-E8h] BYREF
  _BYTE v6[96]; // [rsp+10h] [rbp-E0h] BYREF
  __m128i v7; // [rsp+70h] [rbp-80h] BYREF
  __int64 v8; // [rsp+80h] [rbp-70h]
  char v9; // [rsp+95h] [rbp-5Bh]

  v4 = 0;
  v5 = 0;
  if ( dword_4F08058 )
  {
    sub_771BE0(a1, a2);
    dword_4F08058 = 0;
  }
  sub_774A30((__int64)v6, 1);
  v9 |= 0x10u;
  v8 = *(_QWORD *)(a1 + 28);
  v2 = sub_795190((__int64)v6, a1, (__int64)&v4, &v5);
  *a2 = _mm_loadu_si128(&v7);
  sub_771990((__int64)v6);
  return v2;
}
