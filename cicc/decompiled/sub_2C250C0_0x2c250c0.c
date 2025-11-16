// Function: sub_2C250C0
// Address: 0x2c250c0
//
__int64 __fastcall sub_2C250C0(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v5; // rsi
  __int64 (__fastcall *v6)(__int64 *, __int64); // rax

  v3 = *a2;
  v5 = *a1;
  v6 = *(__int64 (__fastcall **)(__int64 *, __int64))(v3 + 16);
  if ( *a1 )
    v5 = *a1 + 96;
  return v6(a2, v5);
}
