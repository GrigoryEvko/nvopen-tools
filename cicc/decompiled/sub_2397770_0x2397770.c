// Function: sub_2397770
// Address: 0x2397770
//
__int64 __fastcall sub_2397770(__int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 *v4; // r15
  unsigned __int64 *v5; // r13
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rdi

  if ( *(_BYTE *)(a1 + 336) )
  {
    v4 = *(unsigned __int64 **)(a1 + 320);
    v5 = *(unsigned __int64 **)(a1 + 312);
    *(_BYTE *)(a1 + 336) = 0;
    if ( v4 != v5 )
    {
      do
      {
        v6 = v5[1];
        v7 = *v5;
        if ( v6 != *v5 )
        {
          do
          {
            v8 = *(unsigned int *)(v7 + 144);
            v9 = *(_QWORD *)(v7 + 128);
            v7 += 152LL;
            sub_C7D6A0(v9, 8 * v8, 4);
            sub_C7D6A0(*(_QWORD *)(v7 - 56), 8LL * *(unsigned int *)(v7 - 40), 4);
            sub_C7D6A0(*(_QWORD *)(v7 - 88), 16LL * *(unsigned int *)(v7 - 72), 8);
            sub_C7D6A0(*(_QWORD *)(v7 - 120), 16LL * *(unsigned int *)(v7 - 104), 8);
          }
          while ( v6 != v7 );
          v7 = *v5;
        }
        if ( v7 )
          j_j___libc_free_0(v7);
        v5 += 3;
      }
      while ( v4 != v5 );
      v5 = *(unsigned __int64 **)(a1 + 312);
    }
    if ( v5 )
      j_j___libc_free_0((unsigned __int64)v5);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 240), 16LL * *(unsigned int *)(a1 + 256), 8);
  v2 = 16LL * *(unsigned int *)(a1 + 224);
  if ( !*(_DWORD *)(a1 + 224) )
    v2 = 0;
  sub_C7D6A0(*(_QWORD *)(a1 + 208), v2, 8);
  sub_E66D20(a1 + 96);
  sub_B72320(a1 + 96, v2);
  sub_22B0CF0(a1);
  return sub_B72320(a1, v2);
}
