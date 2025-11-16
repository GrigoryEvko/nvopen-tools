// Function: sub_70C860
// Address: 0x70c860
//
__int64 __fastcall sub_70C860(unsigned __int8 a1, const __m128i *a2)
{
  __int64 v2; // rdx
  _BYTE *v4; // rax
  _QWORD v5[2]; // [rsp+0h] [rbp-20h] BYREF
  __int64 v6; // [rsp+10h] [rbp-10h] BYREF
  __int64 v7; // [rsp+18h] [rbp-8h] BYREF

  v5[0] = sub_709B30(a1, a2);
  v6 = v5[0];
  v5[1] = v2;
  v7 = v2;
  if ( unk_4F07580 )
  {
    if ( (HIWORD(v7) & 0x7FFF) != 0x7FFF )
      return (unsigned __int8)sub_12F9B50(v5, &unk_4F07870);
  }
  else if ( (__ROL2__(v6, 8) & 0x7FFF) != 0x7FFF )
  {
    return (unsigned __int8)sub_12F9B50(v5, &unk_4F07870);
  }
  v4 = (char *)&v6 + 2;
  while ( !*v4 )
  {
    if ( (char *)&v7 + 6 == ++v4 )
      return (unsigned __int8)sub_12F9B50(v5, &unk_4F07870);
  }
  return 0;
}
