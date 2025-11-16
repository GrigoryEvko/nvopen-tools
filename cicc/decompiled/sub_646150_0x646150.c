// Function: sub_646150
// Address: 0x646150
//
_BOOL8 __fastcall sub_646150(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // r12

  v1 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)sub_8D3070(v1)
    && (v3 = sub_8D46C0(v1), (unsigned int)sub_8D3CF0(v3))
    && (*(_BYTE *)(v3 + 140) & 0xFB) == 8 )
  {
    return (unsigned int)sub_8D4C10(v3, dword_4F077C4 != 2) == 1;
  }
  else
  {
    return 0;
  }
}
