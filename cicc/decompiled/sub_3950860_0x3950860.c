// Function: sub_3950860
// Address: 0x3950860
//
void __fastcall sub_3950860(_QWORD *a1)
{
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // r14

  v2 = a1[20];
  *a1 = &unk_4A3F1F0;
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 40);
    if ( v3 )
      j_j___libc_free_0(v3);
    v4 = *(unsigned int *)(v2 + 32);
    if ( (_DWORD)v4 )
    {
      v5 = *(_QWORD **)(v2 + 16);
      v6 = &v5[5 * v4];
      do
      {
        if ( *v5 != -16 && *v5 != -8 )
          _libc_free(v5[2]);
        v5 += 5;
      }
      while ( v6 != v5 );
    }
    j___libc_free_0(*(_QWORD *)(v2 + 16));
    j_j___libc_free_0(v2);
  }
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
