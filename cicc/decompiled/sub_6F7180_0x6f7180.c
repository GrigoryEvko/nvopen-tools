// Function: sub_6F7180
// Address: 0x6f7180
//
__int64 __fastcall sub_6F7180(const __m128i *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v8; // rax
  __int64 v9; // rax

  v6 = sub_6F6F40(a1, 0, a3, a4, a5, a6);
  if ( HIDWORD(qword_4F077B4) )
  {
    if ( !a2 )
    {
      v8 = sub_6E8430(v6);
      if ( *(_BYTE *)(v8 + 24) == 1 && (unsigned __int8)(*(_BYTE *)(v8 + 56) - 105) <= 4u )
      {
        v9 = sub_6EE7B0(v8);
        if ( *(_BYTE *)(v9 + 140) == 7
          && (*(_BYTE *)(*(_QWORD *)(v9 + 168) + 20LL) & 4) != 0
          && sub_6E53E0(5, 0x672u, &a1[4].m128i_i32[1]) )
        {
          sub_684B30(0x672u, &a1[4].m128i_i32[1]);
        }
      }
    }
  }
  return v6;
}
