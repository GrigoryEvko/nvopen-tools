// Function: sub_15E4330
// Address: 0x15e4330
//
__int16 __fastcall sub_15E4330(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int16 result; // ax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __m128i v7; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v8[6]; // [rsp+10h] [rbp-30h] BYREF

  sub_15E6420();
  *(_WORD *)(a1 + 18) = *(_WORD *)(a1 + 18) & 0xC00F | *(_WORD *)(a2 + 18) & 0x3FF0;
  *(_QWORD *)(a1 + 112) = *(_QWORD *)(a2 + 112);
  if ( (*(_BYTE *)(a2 + 19) & 0x40) != 0 )
  {
    v2 = sub_15E0FA0(a2);
    v7.m128i_i64[0] = (__int64)v8;
    sub_15DE6F0(v7.m128i_i64, *(_BYTE **)v2, *(_QWORD *)v2 + *(_QWORD *)(v2 + 8));
    sub_15E4280(a1, &v7);
    if ( (_QWORD *)v7.m128i_i64[0] != v8 )
      j_j___libc_free_0(v7.m128i_i64[0], v8[0] + 1LL);
    result = *(_WORD *)(a2 + 18);
    if ( (result & 8) == 0 )
    {
LABEL_5:
      if ( (result & 2) == 0 )
        goto LABEL_6;
      goto LABEL_10;
    }
  }
  else
  {
    sub_15E3BD0(a1);
    result = *(_WORD *)(a2 + 18);
    if ( (result & 8) == 0 )
      goto LABEL_5;
  }
  v4 = sub_15E38F0(a2);
  sub_15E3D80(a1, v4);
  result = *(_WORD *)(a2 + 18);
  if ( (result & 2) == 0 )
  {
LABEL_6:
    if ( (result & 4) == 0 )
      return result;
LABEL_11:
    v6 = sub_15E3950(a2);
    return sub_15E40D0(a1, v6);
  }
LABEL_10:
  v5 = sub_15E3920(a2);
  sub_15E3F20(a1, v5);
  result = *(_WORD *)(a2 + 18);
  if ( (result & 4) != 0 )
    goto LABEL_11;
  return result;
}
