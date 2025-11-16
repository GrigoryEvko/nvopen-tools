// Function: sub_255C2D0
// Address: 0x255c2d0
//
void __fastcall sub_255C2D0(_QWORD *a1)
{
  unsigned __int64 v2; // rbx
  __int64 v3; // r13
  unsigned __int64 v4; // rdi

  v2 = a1[5];
  *a1 = &unk_4A16DD8;
  if ( v2 )
  {
    v3 = (__int64)(a1 + 3);
    do
    {
      sub_255C230(v3, *(_QWORD *)(v2 + 24));
      v4 = v2;
      v2 = *(_QWORD *)(v2 + 16);
      j_j___libc_free_0(v4);
    }
    while ( v2 );
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
