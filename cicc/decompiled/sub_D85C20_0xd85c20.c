// Function: sub_D85C20
// Address: 0xd85c20
//
__int64 __fastcall sub_D85C20(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_D85C20(*(_QWORD *)(v1 + 24));
      v1 = *(_QWORD *)(v1 + 16);
      if ( *(_DWORD *)(v2 + 72) > 0x40u )
      {
        v3 = *(_QWORD *)(v2 + 64);
        if ( v3 )
          j_j___libc_free_0_0(v3);
      }
      if ( *(_DWORD *)(v2 + 56) > 0x40u )
      {
        v4 = *(_QWORD *)(v2 + 48);
        if ( v4 )
          j_j___libc_free_0_0(v4);
      }
      result = j_j___libc_free_0(v2, 80);
    }
    while ( v1 );
  }
  return result;
}
