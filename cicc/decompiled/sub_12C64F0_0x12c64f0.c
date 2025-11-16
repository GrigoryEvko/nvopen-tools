// Function: sub_12C64F0
// Address: 0x12c64f0
//
__int64 __fastcall sub_12C64F0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  _QWORD *v3; // rdi
  _QWORD *v4; // rdi
  _QWORD *v5; // rdi
  _QWORD *v6; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_12C64F0(v1[3]);
      v3 = (_QWORD *)v1[16];
      v1 = (_QWORD *)v1[2];
      if ( v3 != v2 + 18 )
        j_j___libc_free_0(v3, v2[18] + 1LL);
      v4 = (_QWORD *)v2[12];
      if ( v4 != v2 + 14 )
        j_j___libc_free_0(v4, v2[14] + 1LL);
      v5 = (_QWORD *)v2[8];
      if ( v5 != v2 + 10 )
        j_j___libc_free_0(v5, v2[10] + 1LL);
      v6 = (_QWORD *)v2[4];
      if ( v6 != v2 + 6 )
        j_j___libc_free_0(v6, v2[6] + 1LL);
      result = j_j___libc_free_0(v2, 168);
    }
    while ( v1 );
  }
  return result;
}
