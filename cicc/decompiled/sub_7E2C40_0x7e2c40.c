// Function: sub_7E2C40
// Address: 0x7e2c40
//
__int64 __fastcall sub_7E2C40(const __m128i *a1)
{
  __int64 v1; // r13
  __int64 v2; // rax
  __int64 v3; // r12
  char v4; // al
  __int64 v5; // r13
  __m128i *v6; // r12
  _QWORD *v7; // rax

  v1 = sub_72CBE0();
  v2 = sub_7E2C20((__int64)a1);
  if ( v2 )
  {
    v3 = v2;
    v4 = *(_BYTE *)(v2 + 40);
    if ( v4 == 11 )
    {
      sub_7E2C40(v3);
      v5 = *(_QWORD *)(v3 + 48);
    }
    else
    {
      v5 = *(_QWORD *)(v3 + 48);
      if ( !v4 )
      {
        sub_7268E0(v3, 25);
        *(_QWORD *)(v3 + 48) = v5;
      }
    }
    *(_BYTE *)(v5 + 25) &= ~4u;
    v1 = **(_QWORD **)(v3 + 48);
  }
  v6 = (__m128i *)sub_726B30(a1[2].m128i_i8[8]);
  sub_732B40(a1, v6);
  sub_7268E0((__int64)a1, 0);
  v7 = sub_726700(17);
  a1[3].m128i_i64[0] = (__int64)v7;
  *v7 = v1;
  *(_QWORD *)(a1[3].m128i_i64[0] + 56) = v6;
  return sub_7304E0(a1[3].m128i_i64[0]);
}
