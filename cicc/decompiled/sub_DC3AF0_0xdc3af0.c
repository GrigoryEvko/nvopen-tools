// Function: sub_DC3AF0
// Address: 0xdc3af0
//
__int64 __fastcall sub_DC3AF0(__int64 a1, __int64 a2, _BYTE *a3, _BYTE *a4)
{
  __int64 result; // rax
  unsigned int v7; // eax
  char v8; // r8

  if ( (unsigned __int8)sub_DC3A60(a1, a2, a3, a4) )
    return 257;
  v7 = sub_B52870(a2);
  v8 = sub_DC3A60(a1, ((unsigned __int64)BYTE4(a2) << 32) | v7, a3, a4);
  result = 0;
  if ( v8 )
    return 256;
  return result;
}
