// Function: sub_11D2C20
// Address: 0x11d2c20
//
__int64 __fastcall sub_11D2C20(__int64 *a1)
{
  __int64 v1; // r12
  __int64 *v3; // rdi
  __int64 result; // rax

  v1 = *a1;
  if ( *a1 )
  {
    sub_C7D6A0(*(_QWORD *)(v1 + 8), 16LL * *(unsigned int *)(v1 + 24), 8);
    j_j___libc_free_0(v1, 32);
  }
  v3 = (__int64 *)a1[2];
  result = (__int64)(a1 + 4);
  if ( v3 != a1 + 4 )
    return j_j___libc_free_0(v3, a1[4] + 1);
  return result;
}
