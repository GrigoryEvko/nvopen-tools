// Function: sub_2162E20
// Address: 0x2162e20
//
__int64 __fastcall sub_2162E20(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // rsi
  __int64 i; // rbx

  *(_QWORD *)a1 = &unk_4A01FA8;
  if ( *(void **)(a1 + 40) == sub_16982C0() )
  {
    v2 = *(_QWORD *)(a1 + 48);
    if ( v2 )
    {
      v3 = 32LL * *(_QWORD *)(v2 - 8);
      for ( i = v2 + v3; v2 != i; sub_127D120((_QWORD *)(i + 8)) )
        i -= 32;
      j_j_j___libc_free_0_0(v2 - 8);
    }
  }
  else
  {
    sub_1698460(a1 + 40);
  }
  return j_j___libc_free_0(a1, 64);
}
