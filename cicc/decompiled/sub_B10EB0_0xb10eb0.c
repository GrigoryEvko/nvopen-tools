// Function: sub_B10EB0
// Address: 0xb10eb0
//
__int64 __fastcall sub_B10EB0(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // r8d

  v1 = sub_B10CD0(a1);
  v2 = 1;
  if ( v1 )
    return *(_BYTE *)(v1 + 1) >> 7;
  return v2;
}
