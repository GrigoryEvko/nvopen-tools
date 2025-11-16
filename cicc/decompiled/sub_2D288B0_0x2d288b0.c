// Function: sub_2D288B0
// Address: 0x2d288b0
//
void __fastcall sub_2D288B0(__int64 a1)
{
  unsigned __int64 v1; // r13
  unsigned __int64 v2; // r12
  __int64 v3; // rsi

  v1 = *(_QWORD *)a1;
  v2 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v2 )
  {
    do
    {
      v3 = *(_QWORD *)(v2 - 16);
      v2 -= 32LL;
      if ( v3 )
        sub_B91220(v2 + 16, v3);
    }
    while ( v2 != v1 );
    v2 = *(_QWORD *)a1;
  }
  if ( v2 != a1 + 16 )
    _libc_free(v2);
}
