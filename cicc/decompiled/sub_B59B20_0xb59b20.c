// Function: sub_B59B20
// Address: 0xb59b20
//
__int64 __fastcall sub_B59B20(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx

  result = sub_B59AF0(a1);
  if ( !(_BYTE)result )
  {
    v2 = sub_B595C0(a1);
    v3 = sub_ACADE0(*(__int64 ***)(v2 + 8));
    return sub_B59690(a1, v3, v4, v5);
  }
  return result;
}
