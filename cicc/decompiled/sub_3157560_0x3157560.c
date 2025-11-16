// Function: sub_3157560
// Address: 0x3157560
//
void __fastcall sub_3157560(_QWORD *a1)
{
  unsigned __int64 v1; // r13

  v1 = a1[116];
  *a1 = &unk_4A32C90;
  if ( v1 )
  {
    sub_C7D6A0(*(_QWORD *)(v1 + 8), 16LL * *(unsigned int *)(v1 + 24), 8);
    j_j___libc_free_0(v1);
  }
  nullsub_335();
}
