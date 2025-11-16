// Function: sub_2587710
// Address: 0x2587710
//
__int64 __fastcall sub_2587710(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // rax
  unsigned __int8 *v4; // rax
  unsigned __int64 v5; // rsi
  unsigned __int8 v6; // al
  __int64 v7; // rax
  __int64 (__fastcall *v8)(__int64); // rdx
  __int64 v10; // rdi
  unsigned int v11; // r12d
  unsigned __int64 v13[4]; // [rsp+30h] [rbp-20h] BYREF

  v3 = (unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72));
  v4 = sub_98ACB0(v3, 6u);
  if ( !v4 )
    return 0;
  v5 = (unsigned __int64)v4;
  v6 = *v4;
  if ( v6 > 0x1Cu )
  {
    if ( v6 != 60 )
      return 0;
    v10 = *(_QWORD *)(v5 - 32);
    if ( *(_BYTE *)v10 != 17 )
      return 0;
    v11 = *(_DWORD *)(v10 + 32);
    if ( v11 <= 0x40 )
    {
      if ( *(_QWORD *)(v10 + 24) == 1 )
        return *(_QWORD *)(v5 + 72);
    }
    else if ( (unsigned int)sub_C444A0(v10 + 24) == v11 - 1 )
    {
      return *(_QWORD *)(v5 + 72);
    }
    return 0;
  }
  if ( v6 != 22 )
    return 0;
  sub_250D230(v13, v5, 6, 0);
  v7 = sub_2587260(a2, v13[0], v13[1], a1, 0, 0, 1);
  if ( !v7 || !*(_BYTE *)(v7 + 97) )
    return 0;
  v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 112LL);
  if ( v8 == sub_2534DF0 )
    return _mm_loadu_si128((const __m128i *)(v7 + 104)).m128i_u64[0];
  return v8(v7);
}
