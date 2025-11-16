// Function: sub_266E320
// Address: 0x266e320
//
__int64 __fastcall sub_266E320(__int64 *a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v3; // r9
  __int64 v4; // rcx
  unsigned __int8 *v6; // [rsp+8h] [rbp-8h] BYREF

  v3 = *a1;
  v4 = a1[1];
  v6 = a3;
  return (unsigned int)sub_252B2C0(v3, &v6, 1, v4) ^ 1;
}
