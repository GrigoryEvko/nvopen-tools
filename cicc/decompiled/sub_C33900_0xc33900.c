// Function: sub_C33900
// Address: 0xc33900
//
__int64 __fastcall sub_C33900(__int64 a1)
{
  unsigned int v1; // r8d
  __int64 result; // rax

  v1 = sub_C337D0(a1);
  result = a1 + 8;
  if ( v1 > 1 )
    return *(_QWORD *)(a1 + 8);
  return result;
}
