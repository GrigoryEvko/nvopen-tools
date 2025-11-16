// Function: sub_22B07D0
// Address: 0x22b07d0
//
void __fastcall sub_22B07D0(unsigned __int64 *a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r12
  __int64 v4; // rsi
  __int64 v5; // rdi

  v2 = a1[1];
  v3 = *a1;
  if ( v2 != *a1 )
  {
    do
    {
      v4 = *(unsigned int *)(v3 + 144);
      v5 = *(_QWORD *)(v3 + 128);
      v3 += 152LL;
      sub_C7D6A0(v5, 8 * v4, 4);
      sub_C7D6A0(*(_QWORD *)(v3 - 56), 8LL * *(unsigned int *)(v3 - 40), 4);
      sub_C7D6A0(*(_QWORD *)(v3 - 88), 16LL * *(unsigned int *)(v3 - 72), 8);
      sub_C7D6A0(*(_QWORD *)(v3 - 120), 16LL * *(unsigned int *)(v3 - 104), 8);
    }
    while ( v2 != v3 );
    v3 = *a1;
  }
  if ( v3 )
    j_j___libc_free_0(v3);
}
