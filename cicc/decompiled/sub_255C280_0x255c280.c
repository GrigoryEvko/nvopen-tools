// Function: sub_255C280
// Address: 0x255c280
//
void __fastcall sub_255C280(_QWORD *a1)
{
  unsigned __int64 v1; // rbx
  __int64 v2; // r12
  unsigned __int64 v3; // rdi

  v1 = a1[5];
  *a1 = &unk_4A16DD8;
  if ( v1 )
  {
    v2 = (__int64)(a1 + 3);
    do
    {
      sub_255C230(v2, *(_QWORD *)(v1 + 24));
      v3 = v1;
      v1 = *(_QWORD *)(v1 + 16);
      j_j___libc_free_0(v3);
    }
    while ( v1 );
  }
}
