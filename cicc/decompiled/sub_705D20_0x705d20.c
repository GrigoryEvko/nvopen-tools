// Function: sub_705D20
// Address: 0x705d20
//
void __fastcall sub_705D20(_QWORD *a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  __int64 v3; // rdi

  v1 = a1[2];
  while ( v1 )
  {
    v2 = v1;
    sub_705A20(*(_QWORD **)(v1 + 24));
    v3 = *(_QWORD *)(v1 + 32);
    v1 = *(_QWORD *)(v1 + 16);
    if ( v3 != v2 + 48 )
      j_j___libc_free_0(v3, *(_QWORD *)(v2 + 48) + 1LL);
    j_j___libc_free_0(v2, 64);
  }
}
