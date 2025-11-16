// Function: sub_709CC0
// Address: 0x709cc0
//
_BOOL8 __fastcall sub_709CC0(const __m128i *a1, unsigned __int8 a2)
{
  __int64 v2; // rdx
  __int16 v3; // ax
  __int16 v5; // [rsp+0h] [rbp-12h]
  __int64 v6; // [rsp+2h] [rbp-10h]
  __int64 v7; // [rsp+Ah] [rbp-8h]

  v6 = sub_709B30(a2, a1);
  v7 = v2;
  if ( unk_4F07580 )
    v3 = *(__int16 *)((char *)&v5 + n);
  else
    v3 = __ROL2__(v6, 8);
  return (v3 & 0x7FFF) == 0x7FFF;
}
