// Function: sub_67D810
// Address: 0x67d810
//
_BOOL8 __fastcall sub_67D810(unsigned int *a1)
{
  __int64 v1; // rdx
  _BOOL8 result; // rax
  _BYTE v3[4]; // [rsp+8h] [rbp-8h] BYREF
  _BYTE v4[4]; // [rsp+Ch] [rbp-4h] BYREF

  v1 = sub_729B10(*a1, v3, v4, 0);
  result = 0;
  if ( v1 )
    return (*(_BYTE *)(v1 + 72) & 0x40) != 0;
  return result;
}
