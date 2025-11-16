// Function: sub_1E09820
// Address: 0x1e09820
//
__int64 __fastcall sub_1E09820(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  __int64 v3; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_1E09820(v1[3]);
      v3 = v1[5];
      v1 = (_QWORD *)v1[2];
      if ( v3 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 16LL))(v3);
      result = j_j___libc_free_0(v2, 48);
    }
    while ( v1 );
  }
  return result;
}
