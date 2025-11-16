// Function: sub_239FF70
// Address: 0x239ff70
//
void __fastcall sub_239FF70(unsigned __int64 a1)
{
  unsigned __int64 v1; // r14
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // r15
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // [rsp-40h] [rbp-40h]

  if ( a1 )
  {
    v1 = a1;
    do
    {
      sub_239FF70(*(_QWORD *)(v1 + 24));
      v2 = v1;
      v9 = v1;
      v1 = *(_QWORD *)(v1 + 16);
      v3 = *(_QWORD *)(v2 + 112);
      while ( v3 )
      {
        v4 = v3;
        sub_239FEC0(*(_QWORD **)(v3 + 24));
        v5 = *(_QWORD *)(v3 + 96);
        v3 = *(_QWORD *)(v3 + 16);
        while ( v5 )
        {
          v6 = v5;
          sub_239FBF0(*(_QWORD **)(v5 + 24));
          v7 = *(_QWORD *)(v5 + 32);
          v5 = *(_QWORD *)(v5 + 16);
          if ( v7 )
            j_j___libc_free_0(v7);
          j_j___libc_free_0(v6);
        }
        v8 = *(_QWORD *)(v4 + 48);
        if ( v8 != v4 + 64 )
          j_j___libc_free_0(v8);
        j_j___libc_free_0(v4);
      }
      j_j___libc_free_0(v9);
    }
    while ( v1 );
  }
}
