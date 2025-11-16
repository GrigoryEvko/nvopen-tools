// Function: sub_2C0D5F0
// Address: 0x2c0d5f0
//
__int64 __fastcall sub_2C0D5F0(__int64 *a1, __int64 *a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi

  v4 = *a2;
  v5 = *a1;
  if ( *a1 )
    v5 = *a1 + 96;
  return (*(unsigned int (__fastcall **)(__int64 *, __int64))(v4 + 24))(a2, v5) ^ 1;
}
