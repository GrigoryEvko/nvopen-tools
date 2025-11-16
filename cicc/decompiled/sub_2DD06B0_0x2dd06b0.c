// Function: sub_2DD06B0
// Address: 0x2dd06b0
//
void __fastcall sub_2DD06B0(_QWORD *a1)
{
  __int64 v1; // rbx
  unsigned __int64 v2; // r12
  __int64 v3; // rsi
  unsigned __int64 v4; // rdi

  v1 = a1[7];
  v2 = a1[6];
  if ( v1 != v2 )
  {
    do
    {
      v3 = *(_QWORD *)(v2 + 8);
      if ( v3 )
        sub_B91220(v2 + 8, v3);
      v2 += 16LL;
    }
    while ( v1 != v2 );
    v2 = a1[6];
  }
  if ( v2 )
    j_j___libc_free_0(v2);
  v4 = a1[3];
  if ( v4 )
    j_j___libc_free_0(v4);
}
