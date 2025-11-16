// Function: sub_336F5D0
// Address: 0x336f5d0
//
void __fastcall sub_336F5D0(_QWORD *a1)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // rsi
  unsigned __int64 v8; // rdi

  v2 = a1 + 534;
  v3 = a1[532];
  if ( (_QWORD *)v3 != v2 )
    _libc_free(v3);
  v4 = a1[306];
  if ( (_QWORD *)v4 != a1 + 308 )
    _libc_free(v4);
  v5 = a1[240];
  if ( (_QWORD *)v5 != a1 + 242 )
    _libc_free(v5);
  v6 = a1[14];
  if ( (_QWORD *)v6 != a1 + 16 )
    _libc_free(v6);
  v7 = a1[11];
  if ( v7 )
    sub_B91220((__int64)(a1 + 11), v7);
  v8 = a1[7];
  if ( v8 )
    j_j___libc_free_0(v8);
}
