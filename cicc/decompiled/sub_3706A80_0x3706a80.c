// Function: sub_3706A80
// Address: 0x3706a80
//
__int64 __fastcall sub_3706A80(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  volatile signed __int32 *v6; // rdi

  sub_370CE40(a1, *(_QWORD *)(a2 + 8) + 96LL, a3);
  v4 = *(_QWORD **)(a2 + 8);
  *(_QWORD *)(a2 + 8) = 0;
  if ( v4 )
  {
    v5 = v4[14];
    if ( (_QWORD *)v5 != v4 + 16 )
      _libc_free(v5);
    v6 = (volatile signed __int32 *)v4[6];
    v4[4] = &unk_49E6870;
    if ( v6 )
      sub_A191D0(v6);
    j_j___libc_free_0((unsigned __int64)v4);
  }
  return a1;
}
