// Function: sub_39A92A0
// Address: 0x39a92a0
//
__int64 __fastcall sub_39A92A0(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v2; // rbx
  unsigned __int8 *v3; // r15
  __int64 v4; // r13

  v2 = *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8)));
  v3 = sub_39A81B0((__int64)a1, v2);
  v4 = (__int64)sub_39A23D0((__int64)a1, (unsigned __int8 *)a2);
  if ( !v4 )
  {
    v4 = sub_39A5A90((__int64)a1, *(_WORD *)(a2 + 2), (__int64)v3, (unsigned __int8 *)a2);
    sub_39A8AE0(a1, v4, a2);
    sub_39A29E0(a1, v2, a2, v4);
  }
  return v4;
}
