// Function: sub_B2EC90
// Address: 0xb2ec90
//
__int16 __fastcall sub_B2EC90(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _BYTE *v4; // r15
  size_t v5; // r14
  _QWORD *v6; // rax
  __int16 result; // ax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rdi
  size_t v12; // [rsp+18h] [rbp-58h] BYREF
  __m128i v13; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v14[8]; // [rsp+30h] [rbp-40h] BYREF

  sub_B31FB0();
  *(_WORD *)(a1 + 2) = *(_WORD *)(a1 + 2) & 0xC00F | *(_WORD *)(a2 + 2) & 0x3FF0;
  *(_QWORD *)(a1 + 120) = *(_QWORD *)(a2 + 120);
  if ( (*(_BYTE *)(a2 + 3) & 0x40) == 0 )
  {
    sub_B2E730(a1);
    goto LABEL_13;
  }
  v3 = sub_B2DBE0(a2);
  v13.m128i_i64[0] = (__int64)v14;
  v4 = *(_BYTE **)v3;
  v5 = *(_QWORD *)(v3 + 8);
  if ( v5 + *(_QWORD *)v3 )
  {
    if ( !v4 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
  }
  v12 = *(_QWORD *)(v3 + 8);
  if ( v5 > 0xF )
  {
    v13.m128i_i64[0] = sub_22409D0(&v13, &v12, 0);
    v11 = (_QWORD *)v13.m128i_i64[0];
    v14[0] = v12;
  }
  else
  {
    if ( v5 == 1 )
    {
      LOBYTE(v14[0]) = *v4;
      v6 = v14;
      goto LABEL_7;
    }
    if ( !v5 )
    {
      v6 = v14;
      goto LABEL_7;
    }
    v11 = v14;
  }
  memcpy(v11, v4, v5);
  v5 = v12;
  v6 = (_QWORD *)v13.m128i_i64[0];
LABEL_7:
  v13.m128i_i64[1] = v5;
  *((_BYTE *)v6 + v5) = 0;
  sub_B2EBE0(a1, &v13);
  if ( (_QWORD *)v13.m128i_i64[0] != v14 )
  {
    j_j___libc_free_0(v13.m128i_i64[0], v14[0] + 1LL);
    result = *(_WORD *)(a2 + 2);
    if ( (result & 8) != 0 )
      goto LABEL_9;
    goto LABEL_14;
  }
LABEL_13:
  result = *(_WORD *)(a2 + 2);
  if ( (result & 8) != 0 )
  {
LABEL_9:
    v8 = sub_B2E500(a2);
    sub_B2E8C0(a1, v8);
    result = *(_WORD *)(a2 + 2);
    if ( (result & 2) != 0 )
      goto LABEL_10;
LABEL_15:
    if ( (result & 4) != 0 )
      goto LABEL_11;
    return result;
  }
LABEL_14:
  if ( (result & 2) == 0 )
    goto LABEL_15;
LABEL_10:
  v9 = sub_B2E510(a2);
  sub_B2E9C0(a1, v9);
  result = *(_WORD *)(a2 + 2);
  if ( (result & 4) != 0 )
  {
LABEL_11:
    v10 = sub_B2E520(a2);
    return sub_B2EAD0(a1, v10);
  }
  return result;
}
