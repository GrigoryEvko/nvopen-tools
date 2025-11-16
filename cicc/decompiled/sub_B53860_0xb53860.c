// Function: sub_B53860
// Address: 0xb53860
//
__int64 __fastcall sub_B53860(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned int v3; // eax
  char v4; // r8

  if ( (unsigned __int8)sub_B53710(a1, a2) )
    return 257;
  v3 = sub_B52870(a2);
  v4 = sub_B53710(a1, ((unsigned __int64)BYTE4(a2) << 32) | v3);
  result = 0;
  if ( v4 )
    return 256;
  return result;
}
