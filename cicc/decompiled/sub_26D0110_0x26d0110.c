// Function: sub_26D0110
// Address: 0x26d0110
//
void __fastcall sub_26D0110(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r15
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // r13
  __int64 v6; // rax
  unsigned __int64 v7; // r14
  __int64 v8; // rdx
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // [rsp+8h] [rbp-38h]

  if ( a2 > 0x199999999999999LL )
    sub_4262D8((__int64)"vector::reserve");
  v2 = *a1;
  if ( a2 > 0xCCCCCCCCCCCCCCCDLL * ((__int64)(a1[2] - *a1) >> 4) )
  {
    v4 = a1[1];
    v5 = 0;
    v11 = v4 - v2;
    if ( a2 )
    {
      v6 = sub_22077B0(80 * a2);
      v4 = a1[1];
      v2 = *a1;
      v5 = v6;
    }
    if ( v4 != v2 )
    {
      v7 = v5;
      do
      {
        if ( v7 )
        {
          *(_QWORD *)v7 = *(_QWORD *)v2;
          *(_QWORD *)(v7 + 8) = *(_QWORD *)(v2 + 8);
          *(_BYTE *)(v7 + 16) = *(_BYTE *)(v2 + 16);
          *(_BYTE *)(v7 + 17) = *(_BYTE *)(v2 + 17);
          *(_QWORD *)(v7 + 24) = *(_QWORD *)(v2 + 24);
          *(_QWORD *)(v7 + 32) = *(_QWORD *)(v2 + 32);
          *(_QWORD *)(v7 + 40) = *(_QWORD *)(v2 + 40);
          *(_QWORD *)(v7 + 48) = *(_QWORD *)(v2 + 48);
          v8 = *(_QWORD *)(v2 + 56);
          *(_QWORD *)(v2 + 48) = 0;
          *(_QWORD *)(v2 + 40) = 0;
          *(_QWORD *)(v2 + 32) = 0;
          *(_QWORD *)(v7 + 56) = v8;
          *(_QWORD *)(v7 + 64) = *(_QWORD *)(v2 + 64);
          *(_QWORD *)(v7 + 72) = *(_QWORD *)(v2 + 72);
          *(_QWORD *)(v2 + 72) = 0;
          *(_QWORD *)(v2 + 56) = 0;
        }
        else
        {
          v10 = *(_QWORD *)(v2 + 56);
          if ( v10 )
            j_j___libc_free_0(v10);
        }
        v9 = *(_QWORD *)(v2 + 32);
        if ( v9 )
          j_j___libc_free_0(v9);
        v2 += 80LL;
        v7 += 80LL;
      }
      while ( v2 != v4 );
      v2 = *a1;
    }
    if ( v2 )
      j_j___libc_free_0(v2);
    *a1 = v5;
    a1[1] = v5 + v11;
    a1[2] = 80 * a2 + v5;
  }
}
