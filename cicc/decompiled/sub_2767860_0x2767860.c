// Function: sub_2767860
// Address: 0x2767860
//
void __fastcall sub_2767860(unsigned __int64 *a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 *v5; // rdi

  v2 = a1[1];
  v3 = *a1;
  if ( v2 != *a1 )
  {
    do
    {
      if ( *(_DWORD *)(v3 + 88) > 0x40u )
      {
        v4 = *(_QWORD *)(v3 + 80);
        if ( v4 )
          j_j___libc_free_0_0(v4);
      }
      v5 = (unsigned __int64 *)v3;
      v3 += 112LL;
      sub_2767770(v5);
    }
    while ( v2 != v3 );
    v3 = *a1;
  }
  if ( v3 )
    j_j___libc_free_0(v3);
}
