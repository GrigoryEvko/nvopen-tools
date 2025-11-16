// Function: sub_736020
// Address: 0x736020
//
__int64 __fastcall sub_736020(__int64 a1, int a2)
{
  __m128i *v2; // rax
  __int64 v3; // r12
  _BYTE *v4; // rax
  _BYTE *v5; // rsi
  __int64 v6; // rax
  __m128i *v8; // rax
  __int64 v9[2]; // [rsp+8h] [rbp-18h] BYREF

  if ( dword_4F07270[0] == unk_4F073B8 )
  {
    v8 = sub_735FB0(a1, 2, -1);
    v8[11].m128i_i8[0] |= 0x40u;
    v3 = (__int64)v8;
    v5 = sub_735B90(0, (__int64)v8, v9);
  }
  else
  {
    if ( a2 )
      v2 = sub_735FB0(a1, 2, -1);
    else
      v2 = sub_735FB0(a1, 3, -1);
    v2[11].m128i_i8[0] |= 0x40u;
    v3 = (__int64)v2;
    v4 = sub_735B90(dword_4F04C5C, (__int64)v2, v9);
    *(_BYTE *)(v3 + 89) |= 1u;
    v5 = v4;
  }
  sub_72FC40(v3, (__int64)v5);
  v6 = sub_87EBB0(7, 0);
  *(_QWORD *)(v6 + 88) = v3;
  *(_QWORD *)v3 = v6;
  return v3;
}
