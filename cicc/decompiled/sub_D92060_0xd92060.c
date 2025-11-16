// Function: sub_D92060
// Address: 0xd92060
//
__int64 __fastcall sub_D92060(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  _QWORD *v4; // rdi
  __int64 v5; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v2 = a1;
    do
    {
      v3 = v2;
      sub_D92060(v2[3]);
      v4 = (_QWORD *)v2[6];
      v2 = (_QWORD *)v2[2];
      if ( v4 != v3 + 8 )
        _libc_free(v4, a2);
      if ( *((_DWORD *)v3 + 10) > 0x40u )
      {
        v5 = v3[4];
        if ( v5 )
          j_j___libc_free_0_0(v5);
      }
      a2 = 96;
      result = j_j___libc_free_0(v3, 96);
    }
    while ( v2 );
  }
  return result;
}
