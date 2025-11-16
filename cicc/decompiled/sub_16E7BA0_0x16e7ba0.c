// Function: sub_16E7BA0
// Address: 0x16e7ba0
//
__int64 __fastcall sub_16E7BA0(__int64 *a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdx
  __int64 v3; // rax

  v1 = a1[1];
  v2 = a1[3];
  v3 = *a1;
  a1[3] = v1;
  return (*(__int64 (__fastcall **)(__int64 *, __int64, __int64))(v3 + 56))(a1, v1, v2 - v1);
}
