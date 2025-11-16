// Function: sub_621DD0
// Address: 0x621dd0
//
__int64 __fastcall sub_621DD0(__int64 a1)
{
  unsigned int i; // r14d
  __int64 v3; // [rsp+0h] [rbp-80h]
  __int64 v4; // [rsp+8h] [rbp-78h]
  int v5; // [rsp+1Ch] [rbp-64h] BYREF
  __m128i v6; // [rsp+20h] [rbp-60h] BYREF
  __m128i v7; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v8[8]; // [rsp+40h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a1 + 176);
  v3 = *(_QWORD *)(a1 + 184);
  if ( *(__int16 *)(a1 + 176) < 0 && (unsigned int)sub_620E90(a1) )
  {
    sub_620D80(&v6, 0);
    sub_621DB0(&v6);
    v7 = _mm_loadu_si128(&v6);
  }
  else
  {
    sub_620D80(&v6, 1);
    sub_621DB0(&v6);
    sub_620D80(&v7, 0);
  }
  for ( i = 1; ; ++i )
  {
    v8[0] = v4;
    v8[1] = v3;
    sub_6213D0((__int64)v8, (__int64)&v6);
    if ( !(unsigned int)sub_621000((__int16 *)v8, 0, v7.m128i_i16, 0) )
      break;
    sub_621410((__int64)&v6, 1, &v5);
    sub_621410((__int64)&v7, 1, &v5);
  }
  return i;
}
