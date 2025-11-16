// Function: sub_1D13C60
// Address: 0x1d13c60
//
__int64 __fastcall sub_1D13C60(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v2 = *(_QWORD *)(a1 + 16);
    while ( v2 )
    {
      sub_1D13A90(*(_QWORD *)(v2 + 24));
      v3 = v2;
      v2 = *(_QWORD *)(v2 + 16);
      j_j___libc_free_0(v3, 48);
    }
    return j_j___libc_free_0(a1, 48);
  }
  return result;
}
