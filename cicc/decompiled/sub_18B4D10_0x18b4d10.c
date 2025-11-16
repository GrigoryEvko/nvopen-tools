// Function: sub_18B4D10
// Address: 0x18b4d10
//
__int64 **__fastcall sub_18B4D10(__int64 ****a1, __int64 *a2)
{
  __int64 **v2; // rax
  __int64 **v3; // rbx
  __int64 **v4; // r13
  __int64 *v5; // r14

  v2 = (__int64 **)sub_22077B0(24);
  v3 = v2;
  if ( v2 )
    sub_143A950(v2, a2);
  v4 = **a1;
  **a1 = v3;
  if ( v4 )
  {
    v5 = v4[2];
    if ( v5 )
    {
      sub_1368A00(v4[2]);
      j_j___libc_free_0(v5, 8);
    }
    j_j___libc_free_0(v4, 24);
  }
  return **a1;
}
