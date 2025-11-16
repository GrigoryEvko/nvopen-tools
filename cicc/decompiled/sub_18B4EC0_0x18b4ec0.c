// Function: sub_18B4EC0
// Address: 0x18b4ec0
//
__int64 __fastcall sub_18B4EC0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_18B4EC0(v1[3]);
      v3 = v1[11];
      v1 = (_QWORD *)v1[2];
      if ( v3 )
        j_j___libc_free_0(v3, v2[13] - v3);
      v4 = v2[7];
      if ( v4 )
        j_j___libc_free_0(v4, v2[9] - v4);
      v5 = v2[4];
      if ( v5 )
        j_j___libc_free_0(v5, v2[6] - v5);
      result = j_j___libc_free_0(v2, 112);
    }
    while ( v1 );
  }
  return result;
}
