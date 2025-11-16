// Function: sub_2D25520
// Address: 0x2d25520
//
__int64 __fastcall sub_2D25520(_QWORD *a1)
{
  unsigned __int64 v1; // r14
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  __int64 v4; // rsi

  v1 = a1[22];
  *a1 = &unk_4A262E8;
  if ( v1 )
  {
    sub_C7D6A0(*(_QWORD *)(v1 + 120), 16LL * *(unsigned int *)(v1 + 136), 8);
    v2 = *(_QWORD *)(v1 + 56);
    v3 = v2 + 32LL * *(unsigned int *)(v1 + 64);
    if ( v2 != v3 )
    {
      do
      {
        v4 = *(_QWORD *)(v3 - 16);
        v3 -= 32LL;
        if ( v4 )
          sub_B91220(v3 + 16, v4);
      }
      while ( v2 != v3 );
      v3 = *(_QWORD *)(v1 + 56);
    }
    if ( v3 != v1 + 72 )
      _libc_free(v3);
    if ( *(_QWORD *)v1 != v1 + 16 )
      _libc_free(*(_QWORD *)v1);
    j_j___libc_free_0(v1);
  }
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
