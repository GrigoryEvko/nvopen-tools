// Function: sub_F4F5A0
// Address: 0xf4f5a0
//
__int64 __fastcall sub_F4F5A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v2 = a1;
    do
    {
      v3 = v2;
      sub_F4F5A0(*(_QWORD *)(v2 + 24));
      v2 = *(_QWORD *)(v2 + 16);
      if ( *(_BYTE *)(v3 + 104) )
      {
        v4 = *(_QWORD *)(v3 + 48);
        if ( v4 != v3 + 72 )
          _libc_free(v4, a2);
      }
      a2 = 112;
      result = j_j___libc_free_0(v3, 112);
    }
    while ( v2 );
  }
  return result;
}
