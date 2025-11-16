// Function: sub_38DD210
// Address: 0x38dd210
//
__int64 __fastcall sub_38DD210(__int64 a1, int a2)
{
  __int64 result; // rax

  result = sub_38DD140(a1);
  if ( result )
    *(_DWORD *)(result + 76) = a2;
  return result;
}
