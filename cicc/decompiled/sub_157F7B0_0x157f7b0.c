// Function: sub_157F7B0
// Address: 0x157f7b0
//
__int64 __fastcall sub_157F7B0(__int64 a1)
{
  __int64 result; // rax

  result = sub_157ED20(a1);
  if ( *(_BYTE *)(result + 16) != 88 )
    return 0;
  return result;
}
