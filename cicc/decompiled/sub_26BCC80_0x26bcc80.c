// Function: sub_26BCC80
// Address: 0x26bcc80
//
void __fastcall sub_26BCC80(unsigned __int64 a1)
{
  unsigned __int64 v1; // r14
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // r15
  _QWORD *v7; // rdi
  unsigned __int64 v8; // [rsp-40h] [rbp-40h]

  if ( a1 )
  {
    v1 = a1;
    do
    {
      sub_26BCC80(*(_QWORD *)(v1 + 24));
      v2 = v1;
      v8 = v1;
      v1 = *(_QWORD *)(v1 + 16);
      v3 = *(_QWORD *)(v2 + 208);
      while ( v3 )
      {
        v4 = v3;
        sub_26BC990(*(_QWORD **)(v3 + 24));
        v5 = *(_QWORD *)(v3 + 56);
        v3 = *(_QWORD *)(v3 + 16);
        while ( v5 )
        {
          v6 = v5;
          sub_26BCBE0(*(_QWORD **)(v5 + 24));
          v7 = *(_QWORD **)(v5 + 184);
          v5 = *(_QWORD *)(v5 + 16);
          sub_26BC990(v7);
          sub_26BB480(*(_QWORD **)(v6 + 136));
          j_j___libc_free_0(v6);
        }
        j_j___libc_free_0(v4);
      }
      sub_26BB480(*(_QWORD **)(v8 + 160));
      j_j___libc_free_0(v8);
    }
    while ( v1 );
  }
}
