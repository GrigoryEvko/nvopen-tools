// Function: sub_16CE300
// Address: 0x16ce300
//
__int64 __fastcall sub_16CE300(__int64 *a1)
{
  __int64 result; // rax
  _QWORD *v3; // r12
  __int64 v4; // rdi

  result = a1[1];
  v3 = (_QWORD *)(result & 0xFFFFFFFFFFFFFFF8LL);
  if ( (result & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( *v3 )
      j_j___libc_free_0(*v3, v3[2] - *v3);
    result = j_j___libc_free_0(v3, 24);
    a1[1] = 0;
  }
  v4 = *a1;
  if ( *a1 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  return result;
}
