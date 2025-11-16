// Function: sub_108E510
// Address: 0x108e510
//
__int64 (__fastcall **__fastcall sub_108E510(_QWORD *a1))()
{
  __int64 (__fastcall **result)(); // rax
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rdi

  result = off_497C0B0;
  v2 = a1[10];
  *a1 = off_497C0B0;
  if ( v2 )
  {
    v3 = (__int64)(a1 + 8);
    do
    {
      v4 = v2;
      sub_108E240(v3, *(_QWORD **)(v2 + 24));
      v5 = *(_QWORD *)(v2 + 64);
      v2 = *(_QWORD *)(v2 + 16);
      if ( v5 )
        j_j___libc_free_0(v5, *(_QWORD *)(v4 + 80) - v5);
      result = (__int64 (__fastcall **)())j_j___libc_free_0(v4, 88);
    }
    while ( v2 );
  }
  return result;
}
