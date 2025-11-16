// Function: sub_831460
// Address: 0x831460
//
__int64 __fastcall sub_831460(__int64 a1, __int16 a2)
{
  __int64 result; // rax
  __int64 v3; // rax

  if ( (a2 & 1) != 0 && (unsigned int)sub_8D2780(a1) && ((a2 & 0x40) != 0 || !(unsigned int)sub_8D29A0(a1)) )
    return 1;
  if ( (a2 & 0x80u) != 0 && (unsigned int)sub_8D2870(a1) )
    return 1;
  if ( (a2 & 0x100) != 0 && (unsigned int)sub_8D28F0(a1) )
    return 1;
  if ( (a2 & 0x200) != 0 && (unsigned int)sub_8D28B0(a1) )
    return 1;
  if ( (a2 & 0x40) != 0 && (unsigned int)sub_8D29A0(a1) )
    return 1;
  if ( (a2 & 2) != 0 && (unsigned int)sub_8D2A90(a1) )
    return 1;
  if ( (a2 & 4) != 0 && (unsigned int)sub_8D2E30(a1) )
    return 1;
  if ( (a2 & 8) != 0 && (unsigned int)sub_8D2EB0(a1) )
    return 1;
  if ( (a2 & 0x10) != 0 )
  {
    if ( (unsigned int)sub_8D2E30(a1) )
    {
      v3 = sub_8D46C0(a1);
      if ( (unsigned int)sub_8D2310(v3) )
        return 1;
    }
  }
  if ( (a2 & 0x20) != 0 && (unsigned int)sub_8D3D10(a1)
    || (a2 & 0x400) != 0 && (unsigned int)sub_8D3990(a1)
    || (a2 & 0x800) != 0 && (unsigned int)sub_8D3A00(a1) )
  {
    return 1;
  }
  result = a2 & 0x4000;
  if ( (a2 & 0x4000) != 0 )
    return (unsigned int)sub_8D2660(a1) != 0;
  return result;
}
