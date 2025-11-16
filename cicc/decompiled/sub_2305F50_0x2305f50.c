// Function: sub_2305F50
// Address: 0x2305f50
//
void __fastcall sub_2305F50(_QWORD *a1)
{
  unsigned __int64 v1; // r13

  v1 = a1[1];
  *a1 = &unk_4A14BF8;
  if ( v1 )
  {
    sub_C7D6A0(*(_QWORD *)(v1 + 16), 16LL * *(unsigned int *)(v1 + 32), 8);
    j_j___libc_free_0(v1);
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
