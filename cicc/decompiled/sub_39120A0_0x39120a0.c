// Function: sub_39120A0
// Address: 0x39120a0
//
unsigned __int64 __fastcall sub_39120A0(__int64 *a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __m128i v6; // [rsp+0h] [rbp-40h] BYREF
  __int64 v7; // [rsp+10h] [rbp-30h]

  result = sub_38BE350(a1[1]);
  if ( *(_BYTE *)(result + 16) )
  {
    v2 = result;
    v3 = sub_38BFA60(a1[1], 1);
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, v3, 0);
    v4 = *(_QWORD *)(v2 + 8);
    v5 = *(_QWORD *)v2;
    v7 = v3;
    v6.m128i_i64[0] = v5;
    v6.m128i_i32[2] = v4;
    v6.m128i_i16[6] = WORD2(v4);
    v6.m128i_i8[14] = BYTE6(v4);
    *(_BYTE *)(v2 + 16) = 0;
    return sub_3911EC0((_QWORD *)v2, &v6);
  }
  return result;
}
