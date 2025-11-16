// Function: sub_D85F30
// Address: 0xd85f30
//
__int64 __fastcall sub_D85F30(_QWORD *a1)
{
  _QWORD *v1; // r14
  _QWORD *v2; // r13
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_D85F30(v1[3]);
      v3 = v1[17];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = v3;
        sub_D85C20(*(_QWORD *)(v3 + 24));
        v3 = *(_QWORD *)(v3 + 16);
        if ( *(_DWORD *)(v4 + 72) > 0x40u )
        {
          v5 = *(_QWORD *)(v4 + 64);
          if ( v5 )
            j_j___libc_free_0_0(v5);
        }
        if ( *(_DWORD *)(v4 + 56) > 0x40u )
        {
          v6 = *(_QWORD *)(v4 + 48);
          if ( v6 )
            j_j___libc_free_0_0(v6);
        }
        j_j___libc_free_0(v4, 80);
      }
      v7 = v2[11];
      while ( v7 )
      {
        sub_D85A50(*(_QWORD *)(v7 + 24));
        v8 = v7;
        v7 = *(_QWORD *)(v7 + 16);
        j_j___libc_free_0(v8, 40);
      }
      if ( *((_DWORD *)v2 + 16) > 0x40u )
      {
        v9 = v2[7];
        if ( v9 )
          j_j___libc_free_0_0(v9);
      }
      if ( *((_DWORD *)v2 + 12) > 0x40u )
      {
        v10 = v2[5];
        if ( v10 )
          j_j___libc_free_0_0(v10);
      }
      result = j_j___libc_free_0(v2, 168);
    }
    while ( v1 );
  }
  return result;
}
