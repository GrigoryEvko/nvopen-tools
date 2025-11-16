// Function: sub_26C2AF0
// Address: 0x26c2af0
//
void *__fastcall sub_26C2AF0(__int64 a1)
{
  unsigned __int64 v1; // r14
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r15
  unsigned __int64 v5; // r13
  _QWORD *v6; // rdi
  void *result; // rax
  _QWORD *v9; // [rsp+8h] [rbp-38h]

  v9 = *(_QWORD **)(a1 + 16);
  while ( v9 )
  {
    v1 = (unsigned __int64)v9;
    v2 = v9[19];
    v9 = (_QWORD *)*v9;
    while ( v2 )
    {
      v3 = v2;
      sub_26BC990(*(_QWORD **)(v2 + 24));
      v4 = *(_QWORD *)(v2 + 56);
      v2 = *(_QWORD *)(v2 + 16);
      while ( v4 )
      {
        v5 = v4;
        sub_26BCBE0(*(_QWORD **)(v4 + 24));
        v6 = *(_QWORD **)(v4 + 184);
        v4 = *(_QWORD *)(v4 + 16);
        sub_26BC990(v6);
        sub_26BB480(*(_QWORD **)(v5 + 136));
        j_j___libc_free_0(v5);
      }
      j_j___libc_free_0(v3);
    }
    sub_26BB480(*(_QWORD **)(v1 + 104));
    j_j___libc_free_0(v1);
  }
  result = memset(*(void **)a1, 0, 8LL * *(_QWORD *)(a1 + 8));
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  return result;
}
