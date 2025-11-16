// Function: sub_FDD860
// Address: 0xfdd860
//
__int64 __fastcall sub_FDD860(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  _DWORD v4[3]; // [rsp+Ch] [rbp-14h] BYREF

  result = 0;
  v3 = *a1;
  if ( *a1 )
  {
    v4[0] = sub_FDD0F0(*a1, a2);
    return sub_FE8720(v3, v4);
  }
  return result;
}
