// Function: sub_169CA30
// Address: 0x169ca30
//
__int64 __fastcall sub_169CA30(__int64 a1, unsigned __int8 a2)
{
  void **v3; // rdi
  void *v4; // rbx
  void **v5; // rdi

  v3 = (void **)(*(_QWORD *)(a1 + 8) + 8LL);
  v4 = sub_16982C0();
  if ( *v3 == v4 )
    sub_169CA30(v3, a2);
  else
    sub_169B4C0((__int64)v3, a2);
  v5 = (void **)(*(_QWORD *)(a1 + 8) + 40LL);
  if ( *v5 == v4 )
    return sub_169C980(v5, 0);
  else
    return sub_169B620((__int64)v5, 0);
}
