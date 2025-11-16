// Function: sub_3108960
// Address: 0x3108960
//
__int64 __fastcall sub_3108960(__int64 a1)
{
  unsigned int v1; // r8d
  __int64 v2; // rax

  v1 = 21;
  v2 = (unsigned int)(*(_DWORD *)(a1 + 36) - 251);
  if ( (unsigned int)v2 <= 0x1E )
    return dword_44CEF40[v2];
  return v1;
}
