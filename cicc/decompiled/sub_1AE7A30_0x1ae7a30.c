// Function: sub_1AE7A30
// Address: 0x1ae7a30
//
__int64 __fastcall sub_1AE7A30(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  unsigned __int64 v3; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_1AE7A30(*(_QWORD *)(v1 + 24));
      v1 = *(_QWORD *)(v1 + 16);
      if ( *(_BYTE *)(v2 + 96) )
      {
        v3 = *(_QWORD *)(v2 + 48);
        if ( v3 != v2 + 64 )
          _libc_free(v3);
      }
      result = j_j___libc_free_0(v2, 104);
    }
    while ( v1 );
  }
  return result;
}
