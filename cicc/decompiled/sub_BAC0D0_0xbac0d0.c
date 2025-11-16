// Function: sub_BAC0D0
// Address: 0xbac0d0
//
__int64 __fastcall sub_BAC0D0(_QWORD *a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_BAC0D0(v1[3]);
      v3 = v1[8];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        sub_BABF00(*(_QWORD *)(v3 + 24));
        v4 = v3;
        v3 = *(_QWORD *)(v3 + 16);
        j_j___libc_free_0(v4, 48);
      }
      result = j_j___libc_free_0(v2, 96);
    }
    while ( v1 );
  }
  return result;
}
