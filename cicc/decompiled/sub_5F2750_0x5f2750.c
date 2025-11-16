// Function: sub_5F2750
// Address: 0x5f2750
//
__int64 __fastcall sub_5F2750(__int64 a1, __int64 a2, int a3, int a4, unsigned int a5)
{
  __int64 v6; // r12
  char v8; // dl
  __int64 result; // rax

  v6 = a1;
  if ( a3
    && ((unsigned int)sub_8D2930(a1)
     || dword_4F077BC && ((unsigned int)sub_8D2A90(a1) || qword_4F077A8 <= 0x765Bu && (unsigned int)sub_8D2E30(a1))) )
  {
    return 1;
  }
  v8 = *(_BYTE *)(a2 + 172);
  if ( (v8 & 8) != 0 )
  {
    if ( (unsigned int)sub_8D4160(a1) )
      return 1;
    v8 = *(_BYTE *)(a2 + 172);
  }
  if ( (v8 & 0x20) != 0 )
    return 1;
  result = a4 | a5;
  if ( a4 | a5 )
  {
    if ( (v8 & 8) != 0 )
      v6 = sub_8D4130(a1);
    return (unsigned int)sub_8D3D40(v6) != 0;
  }
  return result;
}
