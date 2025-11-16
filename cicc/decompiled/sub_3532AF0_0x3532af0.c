// Function: sub_3532AF0
// Address: 0x3532af0
//
void __fastcall sub_3532AF0(_QWORD *a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  v2 = a1[2];
  v3 = a1[1];
  *a1 = &unk_4A38F98;
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 + 128);
      if ( v4 != v3 + 144 )
        _libc_free(v4);
      v5 = *(_QWORD *)(v3 + 48);
      if ( v5 != v3 + 64 )
        _libc_free(v5);
      v3 += 224LL;
    }
    while ( v2 != v3 );
    v3 = a1[1];
  }
  if ( v3 )
    j_j___libc_free_0(v3);
  j_j___libc_free_0((unsigned __int64)a1);
}
