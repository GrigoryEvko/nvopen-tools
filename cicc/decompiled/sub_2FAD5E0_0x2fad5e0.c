// Function: sub_2FAD5E0
// Address: 0x2fad5e0
//
__int64 __fastcall sub_2FAD5E0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdi
  int v3; // edx

  result = *(_QWORD *)(a1 + 104);
  v2 = a1 + 96;
  if ( v2 != result )
  {
    v3 = 0;
    do
    {
      *(_DWORD *)(result + 24) = v3;
      result = *(_QWORD *)(result + 8);
      v3 += 16;
    }
    while ( v2 != result );
  }
  return result;
}
