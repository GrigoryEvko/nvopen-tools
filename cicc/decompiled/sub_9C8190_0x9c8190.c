// Function: sub_9C8190
// Address: 0x9c8190
//
__int64 *__fastcall sub_9C8190(__int64 *a1, __int64 a2)
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
