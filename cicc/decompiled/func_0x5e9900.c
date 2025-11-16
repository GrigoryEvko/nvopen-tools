// Function: func
// Address: 0x5e9900
//
void __fastcall func(_QWORD *a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdi

  v1 = a1[2];
  while ( v1 )
  {
    sub_5E9730(*(_QWORD *)(v1 + 24));
    v2 = v1;
    v1 = *(_QWORD *)(v1 + 16);
    j_j___libc_free_0(v2, 40);
  }
}
