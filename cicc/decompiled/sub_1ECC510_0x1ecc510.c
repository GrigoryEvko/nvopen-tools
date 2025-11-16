// Function: sub_1ECC510
// Address: 0x1ecc510
//
void __fastcall sub_1ECC510(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12

  v1 = (_QWORD *)a1[2];
  v2 = (_QWORD *)a1[1];
  *a1 = &unk_49FDDA0;
  if ( v1 != v2 )
  {
    do
    {
      if ( *v2 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v2 + 8LL))(*v2);
      ++v2;
    }
    while ( v1 != v2 );
    v2 = (_QWORD *)a1[1];
  }
  if ( v2 )
    j_j___libc_free_0(v2, a1[3] - (_QWORD)v2);
  nullsub_745();
}
