// Function: sub_DCFA10
// Address: 0xdcfa10
//
__int64 __fastcall sub_DCFA10(__int64 *a1, char *a2)
{
  __int64 v2; // rax
  int v3; // eax

  v2 = sub_DCF3A0(a1, a2, 0);
  LOBYTE(v3) = sub_D96A50(v2);
  return v3 ^ 1u;
}
