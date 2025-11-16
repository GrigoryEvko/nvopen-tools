// Function: sub_729F80
// Address: 0x729f80
//
_BOOL8 __fastcall sub_729F80(unsigned int a1)
{
  __int64 v1; // rdx
  _BOOL8 result; // rax
  int v3; // [rsp+8h] [rbp-8h] BYREF
  int v4; // [rsp+Ch] [rbp-4h] BYREF

  v1 = sub_729B10(a1, &v4, &v3, 0);
  result = 0;
  if ( v1 )
    return (*(_BYTE *)(v1 + 72) & 0x40) != 0;
  return result;
}
