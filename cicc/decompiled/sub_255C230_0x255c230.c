// Function: sub_255C230
// Address: 0x255c230
//
void __fastcall sub_255C230(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdi

  if ( a2 )
  {
    v3 = a2;
    do
    {
      sub_255C230(a1, *(_QWORD *)(v3 + 24));
      v4 = v3;
      v3 = *(_QWORD *)(v3 + 16);
      j_j___libc_free_0(v4);
    }
    while ( v3 );
  }
}
