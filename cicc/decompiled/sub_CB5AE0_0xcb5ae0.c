// Function: sub_CB5AE0
// Address: 0xcb5ae0
//
__int64 __fastcall sub_CB5AE0(__int64 *a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdx
  __int64 v3; // rax

  v1 = a1[2];
  v2 = a1[4];
  v3 = *a1;
  a1[4] = v1;
  return (*(__int64 (__fastcall **)(__int64 *, __int64, __int64))(v3 + 72))(a1, v1, v2 - v1);
}
