// Function: sub_274B0A0
// Address: 0x274b0a0
//
unsigned __int64 __fastcall sub_274B0A0(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned __int64 result; // rax
  _BYTE *v5; // rcx

  result = sub_22CE0E0(a3, a1, a2);
  if ( !result && (unsigned __int8)(*(_BYTE *)a1 - 82) <= 1u )
  {
    v5 = *(_BYTE **)(a1 - 32);
    if ( *v5 <= 0x15u )
      return sub_22CF7C0(a3, *(_WORD *)(a1 + 2) & 0x3F, *(_QWORD *)(a1 - 64), (__int64)v5, a2, 0);
  }
  return result;
}
