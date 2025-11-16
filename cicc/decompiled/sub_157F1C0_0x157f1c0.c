// Function: sub_157F1C0
// Address: 0x157f1c0
//
__int64 __fastcall sub_157F1C0(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // r12
  int v3; // ebx
  __int64 result; // rax

  v1 = sub_157EBA0(a1);
  if ( !v1 )
    return 0;
  v2 = v1;
  v3 = sub_15F4D60(v1);
  if ( !v3 )
    return 0;
  result = sub_15F4DF0(v2, 0);
  if ( v3 != 1 )
    return 0;
  return result;
}
