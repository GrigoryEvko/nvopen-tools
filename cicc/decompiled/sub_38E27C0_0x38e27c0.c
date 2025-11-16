// Function: sub_38E27C0
// Address: 0x38e27c0
//
__int64 __fastcall sub_38E27C0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  char v3; // dl
  bool v4; // r8
  unsigned __int64 v5; // rax

  if ( sub_38E27B0(a1) )
  {
    v1 = ((*(unsigned __int16 *)(a1 + 12) >> 3) & 3u) - 1;
    if ( (unsigned int)v1 <= 2 )
      return dword_452E958[v1];
    return 0;
  }
  if ( (*(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    return 0;
  v3 = *(_BYTE *)(a1 + 9);
  if ( (v3 & 0xC) != 8 )
    goto LABEL_8;
  *(_BYTE *)(a1 + 8) |= 4u;
  v5 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a1 + 24));
  *(_QWORD *)a1 = v5 | *(_QWORD *)a1 & 7LL;
  if ( v5 )
    return 0;
  v3 = *(_BYTE *)(a1 + 9);
LABEL_8:
  result = 1;
  if ( (v3 & 2) == 0 )
  {
    v4 = sub_38E2770(a1);
    result = 2;
    if ( !v4 )
      return (unsigned __int8)sub_38E2790(a1) ^ 1u;
  }
  return result;
}
