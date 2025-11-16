// Function: sub_8C7EB0
// Address: 0x8c7eb0
//
__int64 __fastcall sub_8C7EB0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  unsigned int v3; // r14d
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 result; // rax

  v3 = a3;
  if ( (*(_BYTE *)(a1 - 8) & 2) != 0 )
  {
    v5 = sub_8CF970(a1, a3);
    v6 = sub_8CF970(a2, v3);
  }
  else
  {
    v6 = sub_8CF970(a2, a3);
    v5 = sub_8CF970(a1, v3);
  }
  result = v5 == v6;
  if ( v5 != v6
    && a3 == 6
    && (unsigned __int8)(*(_BYTE *)(v5 + 140) - 9) <= 2u
    && (unsigned __int8)(*(_BYTE *)(v6 + 140) - 9) <= 2u )
  {
    return sub_8DA820(v5, v6, 0, 0, 0, 0, 0);
  }
  return result;
}
