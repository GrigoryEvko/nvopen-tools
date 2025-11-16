// Function: sub_B55130
// Address: 0xb55130
//
__int64 __fastcall sub_B55130(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char v3; // r15
  __int64 v4; // r14
  __int16 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r12

  v2 = sub_BD5C60(a1, a2);
  v3 = *(_BYTE *)(a1 + 72);
  v4 = v2;
  v5 = *(_WORD *)(a1 + 2) & 7;
  v6 = sub_BD2C40(80, unk_3F222C8);
  v7 = v6;
  if ( v6 )
    sub_B4D930(v6, v4, v5, v3, 0, 0);
  return v7;
}
