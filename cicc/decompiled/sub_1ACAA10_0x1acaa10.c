// Function: sub_1ACAA10
// Address: 0x1acaa10
//
__int64 __fastcall sub_1ACAA10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v5; // r8d

  result = sub_1ACA9E0(a1, *(unsigned int *)(a2 + 8), *(unsigned int *)(a3 + 8));
  if ( !(_DWORD)result )
  {
    v5 = sub_16A9900(a2, (unsigned __int64 *)a3);
    result = 1;
    if ( v5 <= 0 )
      return (unsigned int)-((int)sub_16A9900(a3, (unsigned __int64 *)a2) > 0);
  }
  return result;
}
