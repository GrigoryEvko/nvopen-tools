// Function: sub_1600BF0
// Address: 0x1600bf0
//
__int64 __fastcall sub_1600BF0(__int64 a1)
{
  __int64 v1; // rax
  char v2; // r15
  __int64 v3; // r14
  int v4; // r13d
  __int64 v5; // rax
  __int64 v6; // r12

  v1 = sub_16498A0(a1);
  v2 = *(_BYTE *)(a1 + 56);
  v3 = v1;
  v4 = (*(unsigned __int16 *)(a1 + 18) >> 1) & 0x7FFFBFFF;
  v5 = sub_1648A60(64, 0);
  v6 = v5;
  if ( v5 )
    sub_15F9C80(v5, v3, v4, v2, 0);
  return v6;
}
