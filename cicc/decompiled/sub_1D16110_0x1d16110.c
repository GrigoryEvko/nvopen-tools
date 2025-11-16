// Function: sub_1D16110
// Address: 0x1d16110
//
bool __fastcall sub_1D16110(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  void *v3; // r13
  __int64 v5; // rdi
  __int64 v6; // rsi

  v2 = *(_QWORD *)(a1 + 88);
  v3 = *(void **)(v2 + 32);
  if ( v3 != *(void **)(a2 + 8) )
    return 0;
  v5 = v2 + 32;
  v6 = a2 + 8;
  if ( v3 == sub_16982C0() )
    return sub_169CB90(v5, v6);
  else
    return sub_1698510(v5, v6);
}
