// Function: sub_1454330
// Address: 0x1454330
//
__int64 __fastcall sub_1454330(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  unsigned __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_1454330(v1[3]);
      v3 = v1[6];
      v1 = (_QWORD *)v1[2];
      if ( (_QWORD *)v3 != v2 + 8 )
        _libc_free(v3);
      if ( *((_DWORD *)v2 + 10) > 0x40u )
      {
        v4 = v2[4];
        if ( v4 )
          j_j___libc_free_0_0(v4);
      }
      result = j_j___libc_free_0(v2, 96);
    }
    while ( v1 );
  }
  return result;
}
