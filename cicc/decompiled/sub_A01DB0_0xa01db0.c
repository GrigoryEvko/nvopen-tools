// Function: sub_A01DB0
// Address: 0xa01db0
//
__int64 *__fastcall sub_A01DB0(__int64 *a1, __int64 a2)
{
  __int64 (__fastcall ***v2)(); // r14
  __int64 v3; // rax
  __int64 v4; // rbx

  v2 = sub_9C8120();
  v3 = sub_22077B0(64);
  v4 = v3;
  if ( v3 )
    sub_C63EB0(v3, a2, 1, v2);
  *a1 = v4 | 1;
  return a1;
}
